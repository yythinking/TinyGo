"""预训练，也用来退火。

    tinygo pretrain                                   # 默认：small，data/pretrain_nano.parquet，写到 out/small/pretrain
    tinygo pretrain --data data/pretrain_mini.parquet --seq-len 2048 --batch-size 16

    # 退火：从预训练的存档出发，九成预训练数据 + 一成对话数据（整段算损失），学习率从 0.1 倍降到 0；给了 --init 默认写到 out/small/anneal
    tinygo pretrain --init out/small/pretrain/ckpt.pt --data data/pretrain_nano.parquet:0.9,data/tokens/chat-all:0.1 \\
        --steps 2000 --warmup-steps 0 --lr-mult 0.1 --final-lr-frac 0

    # WSD：预热、恒定、最后 20% 的步数退火。恒定段的任何存档都能分出一支退火（主线不受影响），写到 out/small/cooldown-<步数>
    tinygo pretrain --schedule wsd
    tinygo pretrain --cooldown-from out/small/pretrain/ckpt.pt --cooldown-steps 3000

--data 可以直接给 parquet，第一次用时自动切词，放在它旁边的 tokens/ 目录里（见 tools.prepare.ensure_tokens）；也可以给切好的目录。

按轮数训（--epochs，默认 1）：数据切成不重叠的片段，每轮打乱一次依次取完，一轮里每个片段恰好看一遍；也可以直接给 --steps。
日志里每一行都写着已经训了多少词（tokens）、多少轮（epoch），不用自己乘。
输出目录里有 config.json（这次的全部参数）、log.jsonl（每隔几步一行）、ckpt.pt（最新的存档）。
中断后加 --resume 接着跑：每一步用哪批数据只由种子和步数决定，所以接着跑和一口气跑完结果一样。
多卡（ZeRO-1，见 training/distributed.py）：用 torchrun 启动，--batch-size 是每张卡的批大小，每步一共 批大小 × 累积 × 卡数 个片段；
这个乘积和单卡一样时，训出来的是同一个模型，存档也能在单卡和多卡之间互相接着训：
    torchrun --standalone --nproc-per-node 8 -m training.train_pretrain --preset medium --data data/tokens/base --batch-size 2
学习率曲线：默认先预热再按余弦降（--schedule cosine）；--schedule wsd 是预热、恒定、最后 --decay-frac 的步数退火（1 − √p 的形状）。
退火分支（--cooldown-from 存档 --cooldown-steps N）：连优化器状态一起从存档接着走，数据也接着主线往后取，
学习率从主线在这一步本来的值，用 N 步降到 --final-lr-frac（不写就和主线一样）。--init 不一样：它只取模型参数，优化器从头开始。
全部参数见文件末尾的 parse_args。
"""
import argparse
import ast
import json
import math
import time
from pathlib import Path

import torch

from lm import PRESETS, LanguageModel, ModelConfig, preset

from .dataset import open_mixed
from .distributed import Zero, init_distributed
from .trainer_utils import (add_optimizer_args, autocast, clip_gradients, compile_model, count, decay_scale, format_eta, load_checkpoint, lr_scale,
                            optimizers_from_args, resume, rows_per_step_of, saturated_share, save_checkpoint, set_lr_scale, setup_device,
                            step_optimizers, wsd_scale)

MODEL_ARGS = ("dim", "n_layers", "n_heads", "n_kv_heads", "max_seq_len", "window", "moe_experts")  # 命令行上能直接改的模型设置
DEFAULTS = dict(data="data/pretrain_nano.parquet", seq_len=2048, batch_size=8, grad_accum=1, seed=0, final_lr_frac=0.1)
# 退火分支不写这几项就取自存档；写了要和存档一样（数据可以换，比如混进对话数据），否则就不是接着主线走了。
# 批大小和累积看的是乘积：批大小 × 累积 × 卡数 要和主线每步的片段数一样（换了卡数时，批大小和累积要自己写）
COOLDOWN_INHERIT = ("data", "seq_len", "batch_size", "grad_accum", "seed")
# 优化器的设置一律取自存档：加载优化器状态时学习率本来就会被存档里的值盖掉
COOLDOWN_OPTIMIZER = ("optimizer", "lr_mult", "matrix_lr", "embed_lr", "head_lr", "engram_lr", "scalar_lr", "weight_decay", "adam_eps", "clip_grad",
                      "muon_momentum", "offload_adam")
# 验证批大小不写（0）就取自存档：抽到的验证片段由它决定，和主线一样，退火分支的验证损失才能和主线直接比
COOLDOWN_VALIDATION = ("eval_batch_size", "eval_micro_batch_size")


def parse_overrides(pairs: list[str]) -> dict:
    """--set 的值：["window=512", "engram_layers=(1,3)"] → 字典，值按 Python 字面量读。"""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--set 要写成 名字=值，收到的是 {pair!r}")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"--set {key} 的值 {value!r} 读不懂") from None
    return out


def schedule_of(args, total: int):
    """学习率曲线：返回「第几步 → 起步值的几倍」。"""
    if getattr(args, "schedule", "cosine") == "wsd":
        decay = max(1, round(args.decay_frac * total))
        return lambda step: wsd_scale(step, total, args.warmup_steps, decay, args.final_lr_frac, args.decay_shape)
    return lambda step: lr_scale(step, total, args.warmup_steps, args.final_lr_frac)


def setup_cooldown(args, world: int = 1) -> tuple[dict, int, float]:
    """--cooldown-from：读主线的存档，把没写的参数补成主线的。返回 (存档, 从第几步分出, 主线在这一步本来的学习率倍数)。"""
    if args.init:
        raise ValueError("--cooldown-from 和 --init 不能一起用：--init 只取模型参数、优化器从头开始，--cooldown-from 连优化器状态一起接着走")
    if not args.cooldown_steps:
        raise ValueError("--cooldown-from 要配 --cooldown-steps：用多少步把学习率降下来")
    path = Path(args.cooldown_from)
    # 先放内存（按需从文件读）：参数和优化器状态加载时才各自搬到设备上，显存里不会同时有两份
    state = load_checkpoint(path, torch.device("cpu"), mmap=True)
    parent = argparse.Namespace(**{"schedule": "cosine", "decay_frac": 0.2, "decay_shape": "sqrt", **state.get("train_config", {})})
    for key in COOLDOWN_INHERIT:
        mine, theirs = getattr(args, key), getattr(parent, key, None)
        if mine is None:
            setattr(args, key, theirs)
        elif key not in ("data", "batch_size", "grad_accum") and mine != theirs:
            raise ValueError(f"--{key.replace('_', '-')} 写的是 {mine}，主线的存档里是 {theirs}。退火分支要接着主线走，这一项要和主线一样（不写就取自存档）")
    rows, theirs = args.batch_size * args.grad_accum * world, rows_per_step_of(vars(parent))
    if theirs is not None and rows != theirs:
        raise ValueError(f"--batch-size × --grad-accum × 卡数 是 {args.batch_size} × {args.grad_accum} × {world} = {rows}，主线每步是 {theirs} 个片段。"
                         f"退火分支要接着主线取数据，每步的片段数要和主线一样（单卡、卡数不变时不写就取自存档）")
    for key in COOLDOWN_OPTIMIZER:
        if hasattr(parent, key):
            setattr(args, key, getattr(parent, key))
    for key in COOLDOWN_VALIDATION:
        if not getattr(args, key):
            setattr(args, key, getattr(parent, key, 0) or 0)
    start = state["step"]
    if "next_lr_scale" in state:
        high = state["next_lr_scale"]
    else:  # 加 WSD 之前存的档：按主线的设置和总步数（存档旁边的 config.json）重算
        config = path.parent / "config.json"
        if not config.exists():
            raise FileNotFoundError(f"{path} 是旧格式的存档，要靠旁边的 {config.name} 算出主线在第 {start} 步的学习率，但它不在")
        high = schedule_of(parent, json.loads(config.read_text(encoding="utf-8"))["steps"])(start)
    if args.final_lr_frac is None:
        args.final_lr_frac = getattr(parent, "final_lr_frac", DEFAULTS["final_lr_frac"])
    args.run = args.run or str(path.parent.parent)
    args.out = args.out or str(Path(args.run) / f"cooldown-{start}")
    return state, start, high


def build_model(args, vocab_size: int, device: torch.device, parent: dict | None = None) -> tuple[LanguageModel, str]:
    """新建模型（预设 + 命令行上改的设置），或者从 --init / --cooldown-from 的存档出发（设置取自存档）。返回模型和一句说明。"""
    changes = {k: getattr(args, k) for k in MODEL_ARGS if getattr(args, k) is not None} | parse_overrides(args.set)
    if args.init or parent is not None:
        if changes:
            raise ValueError(f"从存档出发时模型设置取自存档，不能再改：{'、'.join(changes)}")
        state = parent if parent is not None else load_checkpoint(Path(args.init), torch.device('cpu'), mmap=True)
        model = LanguageModel(ModelConfig.from_json(state["model_config"])).to(device)
        model.load_state_dict(state["model"])
        origin = f"从 {args.init} 出发" if parent is None else f"从 {args.cooldown_from} 第 {state['step']} 步分出的退火分支"
    else:
        model = LanguageModel(preset(args.preset, vocab_size, **changes)).to(device)
        origin = args.preset + ("（改了 " + "、".join(changes) + "）" if changes else "")
    if model.cfg.vocab_size != vocab_size:
        raise ValueError(f"模型的词表是 {model.cfg.vocab_size}，数据是用 {vocab_size} 的词表切的")
    if args.seq_len > model.cfg.max_seq_len:
        raise ValueError(f"片段长度 {args.seq_len} 超过了模型的上限 {model.cfg.max_seq_len}")
    return model, origin


@torch.no_grad()
def evaluate(model, data, args, device: torch.device, zero: Zero | None = None) -> tuple[float, float]:
    """验证集上的平均损失（不含 MoE 的均衡损失）。每次用同一批片段，前后才可比。
    第二个返回值是第一行里 softmax 塌缩的词占比（见 trainer_utils.saturated_share）。多卡时各卡分着算这几批，再合起来。"""
    zero = zero or Zero()
    was_training = model.training
    model.eval()
    total, rows_seen, saturated = 0.0, 0, 0.0
    batch_size = getattr(args, "eval_batch_size", 0) or args.batch_size
    micro_batch_size = getattr(args, "eval_micro_batch_size", 0) or args.batch_size
    for i in range(zero.rank, args.eval_batches, zero.world):
        x, y = data.batch(batch_size, args.seq_len, seed=1_000_003 * (i + 1), device=device)
        for start in range(0, batch_size, micro_batch_size):  # 抽到的片段只由 batch_size 定，微批只决定一次前向几条
            rows = min(micro_batch_size, batch_size - start)
            with autocast(device):
                out = model(x[start:start + rows], y[start:start + rows], return_logits=False)
            total += (out.loss - out.balance_loss).item() * rows
            rows_seen += rows
        if i == 0:
            saturated = saturated_share(model, x, y, device)  # 只看第一行
    model.train(was_training)
    total, rows_seen, saturated = zero.sum_floats(total, rows_seen, saturated)  # 第 0 批在 0 号卡上，别的卡的塌缩比例是 0
    return total / rows_seen, saturated


def total_steps(args, windows: int, world: int = 1) -> int:
    """给了 --steps 就用它；否则按 --epochs 算：每步用 batch_size × grad_accum × 卡数 个片段。"""
    if args.steps:
        return args.steps
    return max(1, math.ceil(args.epochs * windows / (args.batch_size * args.grad_accum * world)))


def tokens_for(spec: str) -> str:
    """把 --data 里的每个 parquet 换成切好的目录（第一次用时切），权重原样保留：「a.parquet:0.9,b:0.1」→「data/tokens/a:0.9,b:0.1」。"""
    from tools.prepare import ensure_tokens

    parts = []
    for item in spec.split(","):
        path, sep, weight = item.rpartition(":")
        if not sep or not weight.replace(".", "", 1).isdigit():
            path, weight = item, ""
        parts.append(ensure_tokens(path) + (f":{weight}" if weight else ""))
    return ",".join(parts)


def train(args) -> dict:
    """跑完训练，返回 {"step", "loss", "val_loss", "tokens"}。用 torchrun 启动时是多卡（ZeRO-1，见 training/distributed.py）。"""
    zero, device = init_distributed(args.device)
    try:
        return run_training(args, zero, device)
    finally:
        zero.close()


def run_training(args, zero: Zero, device: torch.device) -> dict:
    say = print if zero.main else (lambda *a, **k: None)  # 多卡时只有 0 号卡打印、写日志和存档
    args.world_size = zero.world
    if zero.enabled and (args.cuda_graph or args.offload_adam):
        raise ValueError("多卡（ZeRO-1）不支持 --cuda-graph 和 --offload-adam：每张卡本来就只存自己那份优化器状态")
    parent, branch_start, branch_high = setup_cooldown(args, zero.world) if args.cooldown_from else (None, 0, 1.0)
    args.data = zero.main_first(lambda: tokens_for(args.data))  # 第一次用 parquet 时只让 0 号卡切词
    setup_device(device)
    if args.gpu_memory_fraction:
        if device.type != "cuda" or not 0 < args.gpu_memory_fraction <= 1:
            raise ValueError("--gpu-memory-fraction 要求 CUDA，且必须在 0 与 1 之间")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device.index)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 数据
    train_data, val_data = open_mixed(args.data, "train"), open_mixed(args.data, "val")
    if not train_data:
        raise FileNotFoundError(f"{args.data} 里没有训练数据")

    # 模型和优化器
    model, origin = build_model(args, train_data.vocab_size, device, parent)
    model.gradient_checkpointing = device.type == "cuda" and not args.no_checkpoint
    model.checkpoint_policy = args.checkpoint_policy
    zero.setup(model)
    windows = train_data.windows(args.seq_len)
    rows_per_step = args.batch_size * args.grad_accum * zero.world
    optimizers = optimizers_from_args(model, args)
    if args.init_optimizer_state and not args.resume:
        initial=load_checkpoint(Path(args.init),torch.device('cpu'),mmap=True)
        if len(initial['optimizers'])!=len(optimizers):
            raise ValueError('Initialization optimizer count differs')
        for optimizer,saved in zip(optimizers,initial['optimizers']):
            optimizer.load_state_dict(zero.shard(optimizer,saved))
        say(f"Restored optimizer moments from step {initial['step']}; new data and schedule start at zero.")
        del initial
    if parent is not None:  # 退火分支：优化器状态接着主线，学习率从主线这一步的值降到底
        steps = branch_start + args.cooldown_steps
        for opt, opt_state in zip(optimizers, parent["optimizers"]):
            opt.load_state_dict(zero.shard(opt, opt_state))
        schedule = lambda step: decay_scale(step, branch_start, args.cooldown_steps, branch_high, args.final_lr_frac, args.decay_shape)  # noqa: E731
        del parent
    else:
        steps = total_steps(args, windows, zero.world)
        schedule = schedule_of(args, steps)
    if args.resume and not args.eval_batch_size and (out / "ckpt.pt").exists():
        # 接着训时验证批大小不写就沿用存档的：抽到的验证片段由它决定，换了卡数或微批，验证损失也能和前面直接比
        saved = load_checkpoint(out / "ckpt.pt", torch.device("cpu"), mmap=True).get("train_config") or {}
        args.eval_batch_size = saved.get("eval_batch_size") or saved.get("batch_size") or 0
    start_step = (resume(out / "ckpt.pt", model, optimizers, device, rows_per_step, zero.shard) if args.resume else 0) or branch_start
    validation = dict(batch_size=args.eval_batch_size or args.batch_size,
                      micro_batch_size=args.eval_micro_batch_size or args.batch_size,
                      seq_len=args.seq_len, batches=args.eval_batches, seed_multiplier=1_000_003)
    if zero.main:
        (out / "config.json").write_text(json.dumps({"train": vars(args), "steps": steps, "model": json.loads(model.cfg.to_json()), "data": train_data.meta,
                                                    "validation": validation},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    forward = compile_model(model, device, enabled=not args.no_compile, seq_len=args.seq_len, autotune=args.autotune,
                            attention_backend=args.attention_backend)
    graph_step = None
    if args.cuda_graph:
        if device.type != "cuda" or model.cfg.moe_experts:
            raise ValueError("--cuda-graph 要求 CUDA 和普通前馈模型")
        from .cuda_graph import OptimizerGraph
        graph_step = OptimizerGraph(optimizers)
        say("已启用优化器图重放；首步预热和捕获。", flush=True)
    report = model.param_report()
    tokens_per_step = rows_per_step * args.seq_len
    where = f"{device}" + (f" × {zero.world} 个进程（ZeRO-1，每个进程每步 {args.batch_size} × {args.grad_accum} 个片段）" if zero.enabled else "")
    say(f"设备 {where}；模型 {origin}，共 {report['合计'] / 1e6:.1f}M 参数（主干 {report['主干'] / 1e6:.1f}M）；"
        f"数据 {count(train_data.n_tokens)} 词（每轮 {windows} 个片段）；共 {steps} 步 = {count(steps * tokens_per_step)} 词 = "
        f"{steps * rows_per_step / windows:.2f} 轮" + (f"，从第 {start_step} 步接着跑" if start_step else ""), flush=True)
    if args.cooldown_from:
        say(f"学习率在第 {branch_start + 1}～{steps} 步从 {branch_high:.3f} 倍降到 {args.final_lr_frac} 倍（{args.decay_shape}）", flush=True)

    # 训练循环
    model.train()
    smooth, val_loss, loss_value = None, None, float("nan")
    last_time, last_step = time.perf_counter(), start_step
    log = open(out / "log.jsonl", "a", encoding="utf-8") if zero.main else None
    try:
        for step in range(start_step, steps):
            scale = schedule(step)
            set_lr_scale(optimizers, scale * args.lr_mult)
            loss_total = None
            for micro in range(args.grad_accum):  # 攒几个批次的梯度再更新一次
                # 多卡时第 step 步一共取 批大小 × 累积 × 卡数 个片段，和单卡同一步取的正好是同一组
                index = (step * args.grad_accum + micro) * zero.world + zero.rank
                x, y = train_data.ordered_batch(args.batch_size, args.seq_len, index=index, seed=args.seed, device=device)
                with autocast(device):
                    loss = forward(x, y, return_logits=False).loss / args.grad_accum
                loss.backward()
                loss_total = loss.detach() if loss_total is None else loss_total + loss.detach()
            loss_value = zero.mean(loss_total).item()  # 每个完整更新只读回一次，避免逐微批同步。
            if not math.isfinite(loss_value):
                raise RuntimeError(f"第 {step} 步的损失变成了 {loss_value}，训练停下。上一个存档在 {out / 'ckpt.pt'}")
            if zero.enabled:
                zero.reduce_gradients()
                grad_norm = zero.clip_gradients(args.clip_grad)
                zero.sync_balance(model)
                step_optimizers(optimizers)
                zero.broadcast_parameters()
            else:
                grad_norm = clip_gradients(model.parameters(), args.clip_grad)
                if graph_step is None:
                    step_optimizers(optimizers)
                else:
                    graph_step()
            smooth = loss_value if smooth is None else 0.9 * smooth + 0.1 * loss_value

            # 记录、验证、存档
            done = step + 1
            record = None
            if done % args.log_every == 0 or done == steps:
                now = time.perf_counter()
                speed = (done - last_step) * tokens_per_step / max(now - last_time, 1e-9)
                last_time, last_step = now, done
                epoch = done * rows_per_step / windows
                record = dict(step=done, tokens=done * tokens_per_step, epoch=round(epoch, 4), loss=round(loss_value, 4), smooth_loss=round(smooth, 4),
                              lr_scale=round(scale, 4), tokens_per_s=round(speed))
                if grad_norm is not None:
                    record["grad_norm"] = round(float(grad_norm), 4)
                eta = format_eta((steps - done) * tokens_per_step / max(speed, 1e-9))
                say(f"第 {done:>7} 步（{epoch:.3f} 轮，{count(done * tokens_per_step)} 词）  损失 {loss_value:.4f}（平滑 {smooth:.4f}）  "
                    f"学习率×{scale:.3f}" + (f"  梯度 {float(grad_norm):.2f}" if grad_norm is not None else "") +
                    f"  每秒 {speed:,.0f} 词  剩余 {eta}", flush=True)
            if val_data and (done % args.eval_every == 0 or done == steps):
                val_loss, saturated = evaluate(forward, val_data, args, device, zero)
                record = {**(record or dict(step=done)), "val_loss": round(val_loss, 4), "saturated": round(saturated, 6)}
                say(f"第 {done:>7} 步  验证集损失 {val_loss:.4f}  塌缩 {saturated:.2%}", flush=True)
                last_time = time.perf_counter()  # 验证花的时间不算进训练速度
            if record and log:
                log.write(json.dumps(record) + "\n")
                log.flush()
            if done % args.save_every == 0 or done == steps:
                states = zero.gather_optimizer_states(optimizers)  # 多卡时所有卡一起合成，0 号卡写盘
                if zero.main:
                    save_checkpoint(out / "ckpt.pt", model, states, done, train_config=vars(args), total_steps=steps, next_lr_scale=schedule(done))
    finally:
        if log:
            log.close()
    return dict(step=steps, loss=loss_value, val_loss=val_loss, tokens=(steps - start_step) * tokens_per_step)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="预训练（也用来退火）")
    g = p.add_argument_group("数据和输出")
    g.add_argument("--data", default=None,
                   help="parquet（第一次用时自动切词）或切好的目录；可以写成「A:0.9,B:0.1」按比例混合。默认 data/pretrain_nano.parquet，退火分支默认和主线一样")
    g.add_argument("--run", default=None, help="运行目录，各个阶段都放在这里；默认 out/<预设名>")
    g.add_argument("--out", default=None, help="输出目录；默认 <运行目录>/pretrain，给了 --init 时是 <运行目录>/anneal，退火分支是 <运行目录>/cooldown-<步数>")
    g.add_argument("--seq-len", type=int, default=None, help="每个片段多少个词，默认 2048")
    g.add_argument("--batch-size", type=int, default=None, help="每个批次多少个片段，默认 8")

    g = p.add_argument_group("模型（新建时用；--init 时设置取自存档，不能改）")
    g.add_argument("--preset", choices=list(PRESETS), default="small")
    g.add_argument("--dim", type=int, default=None, help="隐藏层宽度")
    g.add_argument("--n-layers", type=int, default=None)
    g.add_argument("--n-heads", type=int, default=None)
    g.add_argument("--n-kv-heads", type=int, default=None)
    g.add_argument("--max-seq-len", type=int, default=None, help="模型能处理的最长序列")
    g.add_argument("--window", type=int, default=None, help="滑动窗口注意力的窗口")
    g.add_argument("--moe-experts", type=int, default=None, help="专家数，0 是不用 MoE")
    g.add_argument("--set", nargs="*", default=[], metavar="名字=值", help="lm/config.py 里其他的设置，比如 engram_layers=(1,5)")
    g.add_argument("--init", default="", help="从这个存档的模型参数出发（退火用），优化器从头开始")
    g.add_argument('--init-optimizer-state',action='store_true',help='Restore optimizer moments from --init while starting the new corpus and schedule at zero')

    g = p.add_argument_group("训练")
    g.add_argument("--epochs", type=float, default=1.0, help="把数据过几遍（可以是小数）；没给 --steps 时按它算")
    g.add_argument("--steps", type=int, default=0, help="一共更新多少次；给了就不看 --epochs")
    g.add_argument("--grad-accum", type=int, default=None, help="攒几个批次的梯度再更新一次（显存不够时用它凑大批次），默认 1")
    g.add_argument("--warmup-steps", type=int, default=100)
    g.add_argument("--final-lr-frac", type=float, default=None, help="学习率最后降到起步值的几分之几，默认 0.1；退火分支默认和主线一样")
    g.add_argument("--seed", type=int, default=None, help="默认 0")
    g.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine", help="cosine：预热后按余弦降；wsd：预热、恒定、最后一段退火")
    g.add_argument("--decay-frac", type=float, default=0.2, help="wsd：最后多大比例的步数用来退火")
    g.add_argument("--decay-shape", choices=["sqrt", "linear"], default="sqrt", help="wsd 和退火分支降学习率的形状：sqrt 是 1 − √p，linear 是直线")

    g = p.add_argument_group("退火分支（从主线的存档分出去退火，主线不受影响）")
    g.add_argument("--cooldown-from", default=None, help="主线的存档；连优化器状态一起接着走，片段长度、批大小、优化器设置都取自它")
    g.add_argument("--cooldown-steps", type=int, default=0, help="用多少步把学习率降到底")
    add_optimizer_args(p)

    g = p.add_argument_group("记录和存档")
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--eval-every", type=int, default=200, help="每隔多少步在验证集上量一次；没有验证集就不量")
    g.add_argument("--eval-batches", type=int, default=20)
    g.add_argument("--eval-batch-size", type=int, default=0, help="验证抽样批大小；0 跟随训练。固定它可跨不同训练微批比较损失")
    g.add_argument("--eval-micro-batch-size", type=int, default=0, help="验证每次前向处理几条；0 跟随训练，不改变抽到的验证片段")
    g.add_argument("--save-every", type=int, default=500)
    g.add_argument("--resume", action="store_true", help="从输出目录里的存档接着跑")

    g = p.add_argument_group("硬件")
    g.add_argument("--device", default="auto", help="auto、cuda、mps 或 cpu")
    g.add_argument("--no-compile", action="store_true", help="CUDA 上也不用 torch.compile")
    g.add_argument('--no-checkpoint', action='store_true', help='Disable activation recomputation when GPU memory permits')
    g.add_argument("--cuda-graph", action="store_true", help="捕获融合 Adam 和 Muon 更新图；只支持 CUDA 普通前馈模型、自定义 Muon")
    g.add_argument("--checkpoint-policy", choices=["selective", "full"], default="selective", help="选择性重算或整层重算")
    g.add_argument("--gpu-memory-fraction", type=float, default=0.0, help="限制 PyTorch CUDA 分配预算；0 不限制，例如 0.85 留出驱动和桌面显存")
    g.add_argument("--autotune", action="store_true", help="为固定形状搜索更快的矩阵和注意力内核；首次启动编译更久")
    g.add_argument("--attention-backend", choices=["flex", "cudnn"], default="flex", help="flex：FlexAttention 窗口内核；cudnn：NVIDIA cuDNN 原生窗口前向和反向（要装 .[cudnn]）")
    args = p.parse_args(argv)
    if args.init_optimizer_state and (not args.init or args.cooldown_from):
        p.error('--init-optimizer-state requires --init and cannot be combined with --cooldown-from')
    if args.eval_batch_size < 0 or args.eval_micro_batch_size < 0:
        p.error("验证批大小必须是正数，或用 0 跟随训练批大小")
    if not args.cooldown_from:  # 退火分支的默认值要读了存档才知道，在 train 里补
        for key, value in DEFAULTS.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        args.run = args.run or f"out/{args.preset}"
        args.out = args.out or str(Path(args.run) / ("anneal" if args.init else "pretrain"))
    return args


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
