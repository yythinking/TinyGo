"""评测一个存档：

    tinygo eval out/demo/sft/ckpt.pt --tasks data/eval_tasks.parquet --judge-url http://localhost:8080/v1 --out out/demo/sft/eval.json

  1. 留出文档（out/tokenizer/heldout.jsonl，分好类别）上每个类别的平均损失：看后训练有没有把预训练学到的东西训坏；
  2. 题目（RL 题目的格式）默认温度 0.8、top-k 50 采样作答，按各自的 reward 打分：数学、代码、工具调用是对错，评审题和诗词题要给评审模型；
  3. 一组固定提问的回答，放进报告里给人看；
  4. 可选：标准基准（--bench ceval cmmlu gsm8k，按对话格式出题，见 eval/benchmarks.py）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from inference.template import ChatFormat
from inference.generate import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, sampling_generator
from inference.tokenizer import DEFAULT_MODEL, Tokenizer
from training.trainer_utils import load_model, pick_device

from .benchmarks import run_benchmarks
from training.rewards.judge import Judge
from training.rewards.tools import ToolEnv
from training.train_rl import score
from training.rollout import run_agent, sample_group

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "out" / "tokenizer" / "heldout.jsonl"
PROMPTS = ["你好，请介绍一下你自己。", "用三句话介绍长江。", "以《秋夜》为题写一首五言绝句。", "把「学而时习之，不亦说乎」翻译成白话。",
           "小明有 12 个苹果，送给同学 5 个，又买了 8 个，现在有几个？", "写一个 Python 函数，判断一个数是不是质数。", "用鲁迅的笔法写一段关于排队的短评。"]


@torch.no_grad()
def heldout_loss(model, tok: Tokenizer, path: Path = HELDOUT, max_len: int | None = None, per_category: int = 50) -> dict:
    """每个类别最多取 per_category 篇，截到模型能处理的长度，算平均每个词的损失。"""
    device = next(model.parameters()).device
    max_len = max_len or model.cfg.max_seq_len
    docs = [json.loads(l) for l in open(path, encoding="utf-8")]
    seen, sums = {}, {}
    model.eval()
    for d in docs:
        cat = d["category"]
        if seen.get(cat, 0) >= per_category:
            continue
        ids = tok.encode(d["text"])[: max_len + 1]
        if len(ids) < 16:
            continue
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        loss = F.cross_entropy(model(x).logits.view(-1, model.cfg.vocab_size), y.view(-1), reduction="sum").item()
        s = sums.setdefault(cat, [0.0, 0])
        s[0] += loss
        s[1] += len(ids) - 1
        seen[cat] = seen.get(cat, 0) + 1
    return {cat: round(total / n, 4) for cat, (total, n) in sorted(sums.items(), key=lambda kv: -kv[1][1])}


@torch.no_grad()
def task_scores(model, fmt: ChatFormat, tasks: list[dict], max_new: int = 256, judge=None,
                temperature: float = DEFAULT_TEMPERATURE, top_k: int | None = DEFAULT_TOP_K, seed: int = 0) -> dict:
    """每道题采样作答一次，按 reward 类型汇总平均分。没有评审模型时跳过 judge、poetry 题。"""
    usable = [t for t in tasks if judge is not None or t["reward"] not in ("judge", "poetry")]
    groups = []
    generator = sampling_generator(next(model.parameters()).device, seed)
    for t in usable:
        if t["reward"] == "agent":
            env = ToolEnv(t)
            msgs = [{"role": "system", "content": env.describe()}] + [m for m in t["messages"] if m["role"] != "system"]
            groups.append([run_agent(model, fmt, msgs, ToolEnv(t), max_new, temperature=temperature, top_k=top_k, generator=generator)])
        else:
            groups.append(sample_group(model, fmt, t["messages"], 1, max_new, temperature=temperature, top_k=top_k, generator=generator))
    rewards = score(usable, groups, judge) if usable else []
    out = {}
    for t, r in zip(usable, rewards):
        out.setdefault(t["reward"], []).append(r[0])
    return {k: dict(n=len(v), mean=round(float(np.mean(v)), 4)) for k, v in out.items()}


@torch.no_grad()
def answers(model, fmt: ChatFormat, prompts: list[str] = PROMPTS, max_new: int = 200,
            temperature: float = DEFAULT_TEMPERATURE, top_k: int | None = DEFAULT_TOP_K, seed: int = 0) -> list[dict]:
    generator = sampling_generator(next(model.parameters()).device, seed)
    return [dict(prompt=p, answer=sample_group(model, fmt, [{"role": "user", "content": p}], 1, max_new,
        temperature=temperature, top_k=top_k, generator=generator)[0].text) for p in prompts]


def run(ckpt: str, tasks: str | None = None, n_tasks: int = 100, judge_url: str = "", judge_model: str = "", max_new: int = 256,
        heldout: Path = HELDOUT, device: str = "auto", out: str | None = None, judge=None, benches=(), bench_limit: int | None = 200,
        bench_dir: Path | None = None, judge_workers: int = 16, judge_timeout: float = 120.0,
        temperature: float = DEFAULT_TEMPERATURE, top_k: int | None = DEFAULT_TOP_K, seed: int = 0) -> dict:
    dev = pick_device(device)
    model = load_model(Path(ckpt), dev).eval()
    tok = Tokenizer(DEFAULT_MODEL)
    fmt = ChatFormat(tok)
    if judge is None and judge_url:
        judge = Judge(judge_url, judge_model or "Qwen/Qwen3.8-27B-FP8", timeout=judge_timeout, workers=judge_workers)
    sampling = dict(temperature=temperature, top_k=top_k, seed=seed)
    report = dict(ckpt=str(ckpt), sampling=sampling)
    if Path(heldout).exists():
        report["heldout_loss"] = heldout_loss(model, tok, Path(heldout))
    if tasks:
        rows, per_kind = [], {}
        for r in pq.read_table(tasks).to_pylist():  # 每种题各取 n_tasks 道，免得文件里排在前面的那种占满
            if per_kind.get(r["reward"], 0) < n_tasks:
                rows.append(r)
                per_kind[r["reward"]] = per_kind.get(r["reward"], 0) + 1
        report["tasks"] = task_scores(model, fmt, rows, max_new, judge, **sampling)
    if benches:
        report["benchmarks"] = run_benchmarks(model, fmt, benches, bench_limit, chat=True, max_new=max_new,
                                              **sampling, **({"bench_dir": bench_dir} if bench_dir else {}))
    report["answers"] = answers(model, fmt, max_new=max_new, **sampling)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="评测一个存档")
    parser.add_argument("ckpt")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--n-tasks", type=int, default=100, help="每种题最多评多少道")
    parser.add_argument("--judge-url", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-workers", type=int, default=16, help="同时发几个评审请求；mlx_lm.server 一次只算一个，用 1–2")
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", nargs="*", default=[], choices=["ceval", "cmmlu", "gsm8k"], help="同时跑这些标准基准")
    parser.add_argument("--bench-limit", type=int, default=200, help="每个基准最多评多少道")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = run(args.ckpt, args.tasks, args.n_tasks, args.judge_url, args.judge_model, args.max_new, device=args.device, out=args.out,
                 benches=args.bench, bench_limit=args.bench_limit, judge_workers=args.judge_workers,
                 temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "answers"}, ensure_ascii=False, indent=1))
    for a in report["answers"]:
        print(f"\n问：{a['prompt']}\n答：{a['answer'][:300]}")
    return report


if __name__ == "__main__":
    main()
