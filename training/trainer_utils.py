"""各个训练脚本（train_*.py）共用的东西：设备和混合精度、学习率曲线、优化器、存档、运行目录。

每个阶段自己的流程（读什么数据、损失怎么算、循环怎么走）都写在它自己的脚本里，这里只放大家都一样的部分。
"""
import contextlib
import math
import os
from pathlib import Path

import torch

from lm import LanguageModel, ModelConfig

from .muon import Muon


# ---------- 设备

def pick_device(name: str = "auto") -> torch.device:
    """auto：有 CUDA 用 CUDA，其次是苹果的 MPS，都没有就用 CPU。"""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def setup_device(device: torch.device) -> None:
    """CUDA 上允许 TF32：矩阵乘法用稍低的内部精度换速度，对训练没有可见影响。"""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def autocast(device: torch.device):
    """CUDA 上前向用 bfloat16（参数仍是单精度）；其他设备不变。"""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def compile_model(model: torch.nn.Module, device: torch.device, enabled: bool = True, seq_len: int | None = None,
                  autotune: bool = False, attention_backend: str = "flex") -> torch.nn.Module:
    """CUDA 上编译前向；激活重计算时逐层编译，保留层间的释放边界。MoE 不编译。"""
    if attention_backend not in ("flex", "cudnn"):
        raise ValueError(f"未知注意力后端：{attention_backend}")
    if attention_backend == "cudnn":
        if device.type != "cuda":
            raise ValueError("cuDNN 注意力要求 CUDA")
        from lm.layers.attention import prepare_cudnn_attention
        prepare_cudnn_attention()
    if device.type == "cuda":
        from lm.loss import LigerFusedLinearCrossEntropyFunction
        if LigerFusedLinearCrossEntropyFunction is None:
            raise RuntimeError("CUDA 加速需要 liger-kernel>=0.8.5；Linux 安装 .[cuda]，Windows 配好 triton-windows 后用 --no-deps 安装 liger-kernel。")
        for block in model.blocks:
            block.attn.backend = attention_backend
            block.attn.prepare(seq_len or block.attn._flex_length or model.cfg.max_seq_len, device)
    if not enabled or device.type != "cuda" or getattr(getattr(model, "cfg", None), "moe_experts", 0):
        return model
    options = {"triton.cudagraphs": False, "max_autotune": autotune}
    if getattr(model, "gradient_checkpointing", False):
        # 整图 AOT 反向会把多层 mHC 的布局转换、重算结果同时保留。
        # checkpoint 放在编译区域外，让每层反向结束后即可释放临时激活。
        from lm.checkpointing import checkpoint_block
        for module in (model.memory, model.entry, model.exit):
            if module is not None:
                module.compile(fullgraph=True, dynamic=False, options=options)
        model.hyper.mappings = torch.compile(model.hyper.mappings, fullgraph=True, dynamic=False, options=options)
        model.checkpoint_block = torch.compile(checkpoint_block, dynamic=False, options=options)
        return model
    return torch.compile(model, dynamic=False, options=options)


@torch.no_grad()
def saturated_share(model, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    """正确答案的领先量超过 FP32 临界值（约 16.6）的词占多少（arXiv 2610.31642 说的 softmax 塌缩）。

    超过这个值，标准交叉熵里正确项的梯度会舍入成 0。只看第一行，整张输出也就「一行词数 × 词表」那么大。
    用未编译的模型，免得编译后的前向为「要输出」再编一遍。"""
    from lm.loss import margin_saturation
    raw = getattr(model, "_orig_mod", model)
    with autocast(device):
        logits = raw(x[:1], return_logits=True).logits
    return margin_saturation(logits, y[:1])


def peak_memory_gb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None


def format_eta(seconds: float) -> str:
    """剩余时间写成人话：45 秒、2 分、3 小时 25 分、2 天 5 小时。算不出来时是「—」。"""
    if not math.isfinite(seconds) or seconds < 0:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s} 秒"
    if s < 3600:
        return f"{s // 60} 分"
    if s < 86400:
        return f"{s // 3600} 小时 {s % 3600 // 60} 分"
    return f"{s // 86400} 天 {s % 86400 // 3600} 小时"


def count(n: float) -> str:
    """词数写成人看的样子：41.9K、1.3M、2.05B。"""
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.3g}{unit}" if n < 100 * size else f"{n / size:,.0f}{unit}"
    return str(int(n))


# ---------- 学习率

def lr_scale(step: int, total_steps: int, warmup_steps: int, final_frac: float = 0.1) -> float:
    """第 step 步（从 0 数）的学习率是起步值的几倍：先线性预热，再按余弦降到 final_frac。WSD 见 wsd_scale。"""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def decay_scale(step: int, start: int, steps: int, high: float, low: float, shape: str = "sqrt") -> float:
    """从第 start 步起，用 steps 步把学习率倍数从 high 降到 low，最后一步（start + steps − 1）正好是 low；start 之前是 high。
    shape：sqrt 按 1 − √p 降（先快后慢；Hägele 等 2024 的比较里比直线好），linear 是直线。"""
    if step < start:
        return high
    p = min(1.0, (step - start + 1) / max(1, steps))
    return high - (high - low) * (math.sqrt(p) if shape == "sqrt" else p)


def wsd_scale(step: int, total_steps: int, warmup_steps: int, decay_steps: int, final_frac: float = 0.1, shape: str = "sqrt") -> float:
    """WSD（预热、恒定、退火）：先线性预热，再一直保持 1，最后 decay_steps 步降到 final_frac。
    和余弦相比，恒定段的任何一个存档都可以分出去单独退火（train_pretrain --cooldown-from），不用事先定死总长度。"""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return decay_scale(step, total_steps - decay_steps, decay_steps, 1.0, final_frac, shape)


def set_lr_scale(optimizers: list[torch.optim.Optimizer], scale: float) -> None:
    """把所有组的学习率设成各自起步值的 scale 倍。"""
    for opt in optimizers:
        for group in opt.param_groups:
            value = group["base_lr"] * scale
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(value)
            else:
                group["lr"] = value


# ---------- 优化器：矩阵始终用 Muon，其余（词向量、输出层、记忆表、归一化系数和零碎的标量）用 Adam

SCALARS = ("pre_logit", "post_logit", "res_logit", "query")  # 形状是矩阵、其实是一组系数的参数
MUON_KINDS = ("custom", "native")  # custom：自己写的（training/muon.py）；native：PyTorch 自带的 torch.optim.Muon（2.9 起）


def split_parameters(model: LanguageModel) -> dict[str, list]:
    """把可训练参数分成五类：embed（词向量）、head（输出层）、engram（记忆表）、scalar（系数和向量）、matrix（矩阵）。"""
    named = {"embed.weight": "embed", "head.weight": "head", "memory.table.weight": "engram"}
    groups = {"embed": [], "head": [], "engram": [], "scalar": [], "matrix": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in named:
            groups[named[name]].append(p)
        elif p.ndim < 2 or name.rsplit(".", 1)[-1] in SCALARS:
            groups["scalar"].append(p)
        else:
            groups["matrix"].append(p)
    return groups


def build_optimizers(model: LanguageModel, kind: str = "custom", matrix_lr: float = 0.02, embed_lr: float = 0.3, head_lr: float = 0.004,
                     engram_lr: float = 0.3, scalar_lr: float = 0.01, weight_decay: float = 0.1, momentum: float = 0.95,
                     adam_betas: tuple = (0.8, 0.95), adam_eps: float = 1e-10, muon_momentum_dtype: torch.dtype | None = None,
                     offload_adam: bool = False) -> list[torch.optim.Optimizer]:
    """返回 [Adam, Muon]，每个参数恰好归一组。Adam 那几组的学习率按 (dim/768)^-0.5 缩放；权重衰减只作用在矩阵上。
    每组记下 base_lr，学习率调度按比例缩放它。这些学习率都是起步值，还没有在我们的数据上调过。

    显存紧时（比如 16 GB 的卡训 large）：muon_momentum_dtype=torch.bfloat16 让 Muon 的动量省一半；
    offload_adam=True 把 Adam 管的那几组（词向量、输出层、记忆表……）的状态放到 CPU 上、在 CPU 上更新（torchao 的 CPUOffloadOptimizer），
    显存里只留参数和梯度。有 MoE 时，Muon 每走一步会顺带调一次路由偏置（model.update_balance）。"""
    if kind not in MUON_KINDS:
        raise ValueError(f"没有 {kind!r} 这种 Muon 路径，可选：{'、'.join(MUON_KINDS)}")
    groups = split_parameters(model)
    scale = (model.cfg.dim / 768) ** -0.5
    spec = [("embed", embed_lr), ("head", head_lr), ("engram", engram_lr), ("scalar", scalar_lr)]
    on_cuda = next(model.parameters()).is_cuda  # CUDA 上用融合版的 Adam：一次调用更新所有参数
    adam_groups = [dict(params=groups[k], lr=lr * scale, name=k) for k, lr in spec if groups[k]]
    if offload_adam:
        if not on_cuda:
            raise RuntimeError("--offload-adam 只在 CUDA 上用：把 Adam 的状态从显存挪到 CPU")
        try:
            from torchao.optim import CPUOffloadOptimizer
        except ImportError:
            raise RuntimeError("--offload-adam 要装 torchao：pip install torchao") from None
        adam = CPUOffloadOptimizer(adam_groups, torch.optim.Adam, betas=adam_betas, eps=adam_eps, weight_decay=0.0)
    else:
        adam = torch.optim.Adam(adam_groups, betas=adam_betas, eps=adam_eps, weight_decay=0.0, fused=on_cuda)
    matrices = [dict(params=groups["matrix"], name="matrix")]
    if kind == "native":
        if muon_momentum_dtype is not None:
            raise ValueError("PyTorch 自带的 Muon 不能改动量的精度；要省显存用 --optimizer custom")
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(f"这版 PyTorch（{torch.__version__}）没有自带的 Muon，要 2.9 以上；或者用 --optimizer custom")
        muon = torch.optim.Muon(matrices, lr=matrix_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True, ns_steps=5,
                                adjust_lr_fn="original")
    else:
        muon = Muon(matrices, lr=matrix_lr, momentum=momentum, weight_decay=weight_decay, momentum_dtype=muon_momentum_dtype)
    if getattr(model.cfg, "moe_experts", 0):
        muon.register_step_post_hook(lambda optimizer, args, kwargs: model.update_balance())
    optimizers = [adam, muon]
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return optimizers


def add_optimizer_args(parser, lr_mult: float = 1.0) -> None:
    """每个训练脚本都有的那组优化器参数。lr_mult 是这个脚本的默认总倍数（SFT、RL 比预训练小）。"""
    g = parser.add_argument_group("优化器（矩阵用 Muon，其余用 Adam；学习率都是起步值，乘上 --lr-mult 和学习率曲线）")
    g.add_argument("--optimizer", choices=MUON_KINDS, default="custom", help="用哪个 Muon：custom 自己写的，native 是 PyTorch 自带的")
    g.add_argument("--lr-mult", type=float, default=lr_mult, help="所有学习率再乘这个数")
    g.add_argument("--matrix-lr", type=float, default=0.02, help="Muon：注意力、前馈、专家、各种投影")
    g.add_argument("--embed-lr", type=float, default=0.3, help="词向量")
    g.add_argument("--head-lr", type=float, default=0.004, help="输出层")
    g.add_argument("--engram-lr", type=float, default=0.3, help="Engram 记忆表（不做权重衰减）")
    g.add_argument("--scalar-lr", type=float, default=0.01, help="归一化系数、多流残差系数、流入口出口的查询向量")
    g.add_argument("--weight-decay", type=float, default=0.1, help="只作用在 Muon 管的矩阵上（解耦：每步先乘 1 − 学习率 × 它）")
    g.add_argument("--adam-eps", type=float, default=1e-10,
                   help="Adam 的 ε。论文 arXiv 2610.31642：ε 越小，正确答案的领先幅度平衡在约 ln(1/ε)；1e-10 时约 23，超过 FP32 的临界值 16.6，"
                        "CUDA 上的融合交叉熵会出现梯度算错的位置；改成 1e-7 能避开，但会改变词向量、输出层、记忆表的学习动态")
    g.add_argument("--clip-grad", type=float, default=1.0, help="梯度裁剪：所有梯度合起来的范数超过它就按比例缩小；0 表示不裁剪")
    g.add_argument("--muon-momentum", choices=["float32", "bfloat16"], default="float32", help="Muon 动量的精度；bfloat16 省一半显存")
    g.add_argument("--offload-adam", action="store_true", help="Adam 那几组的状态放到 CPU 上（要 CUDA 和 torchao），显存紧时用")


def optimizers_from_args(model: LanguageModel, args) -> list[torch.optim.Optimizer]:
    return build_optimizers(model, args.optimizer, matrix_lr=args.matrix_lr, embed_lr=args.embed_lr, head_lr=args.head_lr,
                            engram_lr=args.engram_lr, scalar_lr=args.scalar_lr, weight_decay=args.weight_decay, adam_eps=args.adam_eps,
                            muon_momentum_dtype=torch.bfloat16 if args.muon_momentum == "bfloat16" else None, offload_adam=args.offload_adam)


def clip_gradients(params, max_norm: float) -> torch.Tensor | None:
    """把梯度的总范数裁到 max_norm 以内，返回裁剪前的总范数（记进日志）；max_norm 为 0 时不裁、返回 None。"""
    if not max_norm:
        return None
    params = list(params)
    # foreach 是一次处理一批张量的快速实现；MPS 上 PyTorch 不支持，硬要用会报错，交给 PyTorch 自己选
    return torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True,
                                         foreach=True if params and params[0].is_cuda else None)


def step_optimizers(optimizers: list[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)


def rng_state() -> dict:
    """Save sampling/dropout state without initializing CUDA in a CPU process."""
    import random
    import numpy as np
    result = dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())
    if torch.cuda.is_initialized():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state: dict | None) -> None:
    if not state:
        return
    import random
    import numpy as np
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


# ---------- 运行目录：一次完整训练的各个阶段都放在同一个目录下（默认 out/small），每个阶段写到 <运行目录>/<阶段>/ckpt.pt，
# 下一个阶段不写 --init 时，就从这里找上一个阶段的存档。

STAGE_COMMANDS = {"pretrain": "tinygo pretrain（也就是 python -m training.train_pretrain）",
                  "anneal": "tinygo pretrain --init <预训练的存档>（也就是 python -m training.train_pretrain）",
                  "sft": "tinygo sft（也就是 python -m training.train_sft）", "mopd": "tinygo distill（也就是 python -m training.train_distill）",
                  "rl": "tinygo rl（也就是 python -m training.train_rl）"}


def find_stage(run, stages: tuple) -> Path:
    """按顺序找 <run>/<阶段>/ckpt.pt，返回第一个存在的；都没有就报错，说明该先跑哪一步。"""
    for stage in stages:
        path = Path(run) / stage / "ckpt.pt"
        if path.exists():
            return path
    tried = "、".join(f"{stage}/ckpt.pt" for stage in stages)
    hint = STAGE_COMMANDS.get(stages[-1], f"{stages[-1]} 这个阶段")
    raise FileNotFoundError(f"在 {run} 下没找到 {tried}：先跑 {hint}，或者用 --init 指定存档（--run 可以换运行目录）")


# ---------- 存档：模型参数、优化器状态、走到第几步、模型配置。先写临时文件再改名，中途断电不会留下半个存档。

def save_checkpoint(path: Path, model: LanguageModel, optimizers: list, step: int, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # optimizers 里可以是优化器，也可以是已经取好的状态（多卡时由 Zero.gather_optimizer_states 合成的）
    states = [opt if isinstance(opt, dict) else opt.state_dict() for opt in optimizers]
    state = dict(model=model.state_dict(), optimizers=states, step=step, model_config=model.cfg.to_json(), **extra)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device, mmap: bool = False) -> dict:
    """mmap=True：张量按需从文件里读，不一次全读进内存（多卡时每个进程都要读同一个存档）。"""
    path = Path(path)
    if path.is_dir() or path.suffix == '.safetensors':
        from inference.artifact import read_artifact
        return read_artifact(path, device=device)
    return torch.load(path, map_location=device, weights_only=False, mmap=mmap)


def load_model(path: Path, device: torch.device) -> LanguageModel:
    """只要模型（接着训练、生成、评测时用）。"""
    state = load_checkpoint(path, torch.device('cpu'), mmap=True)
    model = LanguageModel(ModelConfig.from_json(state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    return model


def rows_per_step_of(config: dict) -> int | None:
    """存档里记的训练设置每步用多少个片段：批大小 × 累积 × 卡数（以前的存档没记卡数，就是单卡）。"""
    if not config.get("batch_size"):
        return None
    return config["batch_size"] * (config.get("grad_accum") or 1) * (config.get("world_size") or 1)


def resume(path: Path, model: LanguageModel, optimizers: list, device: torch.device, rows_per_step: int | None = None, shard=None) -> int:
    """从 path 的存档接着训：恢复模型和优化器，返回已经走完的步数；没有存档返回 0。
    rows_per_step：这次每步多少个片段。和存档时不一样就拒绝：每步取哪些片段由步数决定，每步的片段数变了，数据就接不上了。
    shard：多卡时只留这张卡管的那部分优化器状态（Zero.shard）。参数和状态先读到内存，加载时再各自搬到参数所在的设备上。"""
    if not Path(path).exists():
        return 0
    state = load_checkpoint(path, torch.device("cpu"), mmap=True)
    saved = rows_per_step_of(state.get("train_config") or {})
    if rows_per_step is not None and saved is not None and saved != rows_per_step:
        raise ValueError(f"存档每步 {saved} 个片段，这次每步 {rows_per_step} 个（批大小 × 累积 × 卡数）。"
                         f"每步取哪些片段由步数决定，要接着训，这个乘积要和存档一样，比如 8 张卡就用 --batch-size {max(1, saved // 8)} --grad-accum 1")
    if len(state.get('optimizers',[]))!=len(optimizers):
        raise ValueError('Resuming requires the full optimizer checkpoint; a model-only checkpoint can only initialize a new stage')
    model.load_state_dict(state["model"])
    for opt, opt_state in zip(optimizers, state["optimizers"]):
        opt.load_state_dict(shard(opt, opt_state) if shard else opt_state)
    return state["step"]
