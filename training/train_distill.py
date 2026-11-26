"""MOPD（多老师在线蒸馏）：学生自己生成回答，按每道题的领域找对应的专家当老师，在学生写的每个位置上最小化 KL(学生 ‖ 老师)。

    tinygo distill        # 默认：学生是 out/small/sft，老师是 out/small 里所有的 expert_<领域>，data/mopd_prompts.parquet，写到 out/small/mopd
    tinygo distill --run out/medium --teachers 代码=out/medium/expert_code/ckpt.pt 数学=out/medium/expert_math/ckpt.pt

专家和学生用同一个分词器，所以 KL 是在整个词表上精确算的，不是只靠采样出的那一个词去估。只有一个老师时就是普通的在线蒸馏。
--merge-init：先把学生换成「SFT 模型 + 各专家相对它的改动的平均」（任务算术，CA-MOPD 的做法），再开始蒸馏。

提示数据（parquet，一行一道）：messages（不含回答）、domain。domain 有同名的老师就交给它；没有就按路由表 ROUTES 找
（比如逻辑题交给数学专家、阅读和诗词交给文学专家），--route 领域=老师 可以改、可以加。
全部参数见文件末尾的 parse_args。
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from inference.template import ChatFormat
from inference.tokenizer import DEFAULT_MODEL, Tokenizer

from .rollout import collate, keep_fitting, sample_group
from .trainer_utils import (add_optimizer_args, autocast, clip_gradients, find_stage, format_eta, load_checkpoint, load_model,
                            optimizers_from_args, pick_device, resume, restore_rng, rng_state,
                            save_checkpoint, set_lr_scale, setup_device, step_optimizers)


# 没有单独专家的领域交给哪位老师
EXPERT_DOMAINS = {"math": "数学", "code": "代码", "literature": "文学"}  # 运行目录里 expert_<名字> 是哪个领域的老师（和 dataset/sft.py 的 EXPERTS 对应）
ROUTES = {"逻辑": "数学", "阅读": "文学", "诗词": "文学", "写作": "文学", "古文": "文学", "指令跟随": "通用", "科学": "通用", "工具调用": "通用"}


def route(domain: str, teachers, routes: dict) -> str | None:
    """这个领域的题交给哪位老师；找不到返回 None。"""
    if domain in teachers:
        return domain
    to = routes.get(domain)
    return to if to in teachers else None


def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """每个位置上的 KL(学生 ‖ 老师) = Σ p学生 (log p学生 − log p老师)，(B, T)。"""
    s = F.log_softmax(student_logits.float(), dim=-1)
    t = F.log_softmax(teacher_logits.float(), dim=-1)
    return (s.exp() * (s - t)).sum(-1)


@torch.no_grad()
def merge_into(student, experts: list) -> None:
    """任务算术：学生 ← 学生 + 平均(专家 − 学生)。"""
    params = dict(student.named_parameters())
    for name, p in params.items():
        delta = sum(dict(e.named_parameters())[name] - p for e in experts) / len(experts)
        p.add_(delta)


def train(cfg) -> dict:
    """跑完蒸馏，返回最后一步的统计。cfg 是 parse_args 的结果（teachers 已经是 {领域: 存档}）。"""
    device = pick_device(cfg.device)
    setup_device(device)
    if cfg.gpu_memory_fraction:
        if device.type!='cuda' or not 0<cfg.gpu_memory_fraction<=1:raise ValueError('Invalid CUDA memory fraction')
        torch.cuda.set_per_process_memory_fraction(cfg.gpu_memory_fraction,device.index)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and any((out / name).exists() for name in ('ckpt.pt', 'log.jsonl')):
        raise ValueError('Distillation output contains a run; use --resume or a new --out')
    if cfg.resume and not (out / 'ckpt.pt').exists():
        raise FileNotFoundError('--resume requires a distillation checkpoint')
    if not cfg.teachers:
        raise ValueError("至少要有一个老师：--teachers 领域=存档")
    prompts = pq.read_table(cfg.prompts).to_pylist()
    routes = {**ROUTES, **cfg.routes}
    for p in prompts:
        p["teacher"] = route(p.get("domain"), cfg.teachers, routes)
    unknown = sorted({p.get("domain") for p in prompts if p["teacher"] is None})
    if unknown:
        raise ValueError(f"这些领域没有对应的老师：{unknown}；老师有 {sorted(cfg.teachers)}，可以用 --route 领域=老师 指定")
    student = load_model(Path(cfg.init), device)
    student.gradient_checkpointing = device.type == 'cuda'
    student.checkpoint_policy = 'selective'
    teachers = {d: load_model(Path(p), device).eval() for d, p in cfg.teachers.items()}
    for network in [student, *teachers.values()]:
        for block in network.blocks:
            block.attn.backend = 'sdpa'
            block.attn.fuse_qkv = False
    for name, t in teachers.items():
        if t.cfg.vocab_size != student.cfg.vocab_size:
            raise ValueError(f"老师 {name} 的词表和学生不一样，没法算全词表 KL")
        t.requires_grad_(False)
    if cfg.merge_init and not cfg.resume:
        merge_into(student, list(teachers.values()))
    fmt = ChatFormat(Tokenizer(DEFAULT_MODEL))
    prompts = keep_fitting(student, fmt, prompts, cfg.max_new, "MOPD 提示")
    optimizers = optimizers_from_args(student, cfg)
    start_step = resume(out / 'ckpt.pt', student, optimizers, device) if cfg.resume else 0
    (out / "config.json").write_text(json.dumps({"distill": vars(cfg),"steps":cfg.steps, "model": json.loads(student.cfg.to_json())}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed) if device.type in ('cpu', 'cuda') else None
    if cfg.resume:
        state = load_checkpoint(out / 'ckpt.pt', torch.device('cpu'), mmap=True)
        previous = state.get('distill_config', {})
        for key in ('init', 'teachers', 'routes', 'prompts', 'seed', 'prompts_per_step', 'samples_per_prompt', 'max_new', 'temperature','eager_rollout'):
            if previous.get(key,False if key=='eager_rollout' else None) != getattr(cfg, key):
                raise ValueError(f'Distillation resume changes its sampling contract: {key}')
        progress = state['distill_progress']
        rng.bit_generator.state = progress['sample_rng']
        if gen is not None:
            gen.set_state(progress['token_rng'])
        restore_rng(state.get('rng'))
        del state
    counts = {}
    for p in prompts:
        counts[p["teacher"]] = counts.get(p["teacher"], 0) + 1
    print(f"设备 {device}；学生 {cfg.init}；{len(prompts)} 道提示，交给各位老师：" + "，".join(f"{t} {n}" for t, n in counts.items()) + f"；共 {cfg.steps} 步", flush=True)
    stats, started = {}, time.perf_counter()
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        for step in range(start_step, cfg.steps):
            t0 = time.perf_counter()
            set_lr_scale(optimizers, cfg.lr_mult * min(1.0, (step + 1) / max(1, cfg.warmup_steps)))
            batch = [prompts[i] for i in rng.choice(len(prompts), size=min(cfg.prompts_per_step, len(prompts)), replace=False)]
            by_domain = {}
            for p in batch:
                with autocast(device):
                    by_domain.setdefault(p["teacher"], []).extend(sample_group(student, fmt, p["messages"], cfg.samples_per_prompt, cfg.max_new,
                                                                              cfg.temperature, None, gen,fast=False if cfg.eager_rollout else None))
            n_tokens = sum(sum(e.learn) for eps in by_domain.values() for e in eps)
            if not n_tokens:raise ValueError('MOPD rollout contains no response tokens')
            student.train()
            total, per_domain = 0.0, {}
            for domain, episodes in by_domain.items():
                kl_sum = 0.0
                for start in range(0, len(episodes), cfg.micro_batch):
                    x, y, learn = collate(episodes[start:start + cfg.micro_batch], device)
                    with autocast(device):
                        s_logits = student(x).logits
                        with torch.no_grad():
                            t_logits = teachers[domain](x).logits
                    # Prompt and padding positions have no objective. Select
                    # response rows before allocating FP32 vocabulary tensors.
                    kl = reverse_kl(s_logits[learn], t_logits[learn])
                    loss = kl.sum() / max(1, n_tokens)
                    loss.backward()
                    kl_sum += kl.sum().item()
                    del s_logits,t_logits,kl,loss
                total += kl_sum
                per_domain[domain] = round(kl_sum / max(1, sum(sum(e.learn) for e in episodes)), 4)
            if not math.isfinite(total):
                raise RuntimeError(f"第 {step} 步的 KL 变成了 {total}")
            clip_gradients(student.parameters(), cfg.clip_grad)
            step_optimizers(optimizers)
            stats = dict(step=step + 1, kl=round(total / max(1, n_tokens), 4), by_domain=per_domain, response_tokens=n_tokens,
                         lr_scale=min(1.0,(step+1)/max(1,cfg.warmup_steps)),lr_mult=cfg.lr_mult,step_s=round(time.perf_counter() - t0, 1))
            log.write(json.dumps(stats, ensure_ascii=False) + "\n")
            log.flush()
            if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
                eta = format_eta((cfg.steps - step - 1) * (time.perf_counter() - started) / (step + 1 - start_step))
                print(f"第 {step + 1:>5} 步  KL {stats['kl']:.4f}（" + "，".join(f"{d} {v:.3f}" for d, v in per_domain.items()) +
                      f"）  {stats['step_s']}s  剩余 {eta}", flush=True)
            if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
                save_checkpoint(out / 'ckpt.pt', student, optimizers, step + 1, distill_config=vars(cfg), rng=rng_state(),
                                distill_progress=dict(sample_rng=rng.bit_generator.state,
                                                     token_rng=gen.get_state() if gen is not None else None))
    if cfg.steps:
        save_checkpoint(out/'last-model.pt',student,[],cfg.steps,distill_config=vars(cfg),
                        checkpoint_role='model_only_completed_stage',resume_supported=False)
    (out/'summary.json').write_text(json.dumps(dict(step=cfg.steps,stats=stats,last_model='last-model.pt',
                                                   teachers=cfg.teachers),ensure_ascii=False,indent=2),encoding='utf-8')
    return stats


def parse_routes(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        domain, sep, teacher = pair.partition("=")
        if not sep or not domain or not teacher:
            raise ValueError(f"--route 要写成 领域=老师，收到的是 {pair!r}")
        out[domain] = teacher
    return out


def find_teachers(run) -> list[str]:
    """运行目录里的专家存档（expert_<名字>/ckpt.pt），写成 领域=存档。"""
    pairs = [f"{domain}={Path(run) / f'expert_{name}' / 'ckpt.pt'}" for name, domain in EXPERT_DOMAINS.items()
             if (Path(run) / f"expert_{name}" / "ckpt.pt").exists()]
    if not pairs:
        raise FileNotFoundError(f"在 {run} 下没找到专家的存档（expert_math、expert_code、expert_literature）："
                                f"先用 tinygo sft --data data/sft_expert_math.parquet 这样训出专家，或者用 --teachers 指定")
    return pairs


def parse_teachers(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        domain, sep, path = pair.partition("=")
        if not sep or not domain or not path:
            raise ValueError(f"--teachers 要写成 领域=存档，收到的是 {pair!r}")
        out[domain] = path
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MOPD：多老师在线蒸馏")
    g = p.add_argument_group("模型、提示和输出")
    g.add_argument("--run", default="out/small", help="运行目录，各个阶段都放在这里")
    g.add_argument("--init", default=None, help="学生从这个存档出发（通用 SFT 模型）；默认 <运行目录>/sft/ckpt.pt")
    g.add_argument("--prompts", default="data/mopd_prompts.parquet", help="提示数据（parquet），每行有 messages 和 domain")
    g.add_argument("--out", default=None, help="输出目录；默认 <运行目录>/mopd")
    g.add_argument("--teachers", nargs="+", default=None, metavar="领域=存档",
                   help="每个领域的老师，比如 数学=out/small/expert_math/ckpt.pt；默认是运行目录里所有的 expert_<领域>")
    g.add_argument("--route", nargs="*", default=[], metavar="领域=老师", help="没有同名老师的领域交给谁，在默认路由表（ROUTES）上改或加")
    g.add_argument("--merge-init", action="store_true", help="先把学生换成 SFT 模型 + 各专家改动的平均（任务算术），再开始蒸馏")

    g = p.add_argument_group("采样")
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--prompts-per-step", type=int, default=8)
    g.add_argument("--samples-per-prompt", type=int, default=4)
    g.add_argument("--max-new", type=int, default=512)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--micro-batch", type=int, default=8)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument('--eager-rollout',action='store_true',help='Use ordinary decoding for short diagnostics')

    g = p.add_argument_group("训练")
    g.add_argument("--warmup-steps", type=int, default=5)
    add_optimizer_args(p, lr_mult=0.05)

    g = p.add_argument_group("记录、存档和硬件")
    g.add_argument("--log-every", type=int, default=1)
    g.add_argument("--save-every", type=int, default=50)
    g.add_argument("--device", default="auto")
    g.add_argument('--gpu-memory-fraction',type=float,default=0.)
    g.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    args.init = args.init or str(find_stage(args.run, ("sft",)))
    args.out = args.out or str(Path(args.run) / "mopd")
    args.teachers = parse_teachers(args.teachers or find_teachers(args.run))
    args.routes = parse_routes(args.route)
    return args


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
