"""过拟合检查：拿一小段真实文字反复训，看模型能不能把它背下来。整条链（分词 → 训练 → 存档 → 生成）都走一遍。

    tinygo overfit --preset small --chars 3000 --steps 300

做法：
  1. 从 --input（默认是洗好的鲁迅全集）里取前 --chars 个字；
  2. 用正式分词器切词，写成一个临时的词数据目录（训练集和验证集是同一段，验证集损失就是背得怎么样）；
  3. 用 training.train_pretrain 在上面训 --steps 步；
  4. 从开头 --prompt-tokens 个词开始贪心续写，逐词和原文对比，报告续写对上了多少。

损失降到 0.1 以下、续写几乎全对，说明模型、优化器、训练循环都是通的。拿到新机器先跑一次。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from inference.generate import generate
from inference.tokenizer import DEFAULT_MODEL, Tokenizer
from lm import PRESETS
from training import train_pretrain
from training.dataset import ShardWriter, write_meta
from training.trainer_utils import load_model, pick_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "clean" / "luxun"


def load_text(path: Path, chars: int) -> str:
    """从一个 parquet 文件或一个目录里的 parquet 文件，按顺序取 text 列，凑够 chars 个字。"""
    files = sorted(Path(path).glob("*.parquet")) if Path(path).is_dir() else [Path(path)]
    pieces, total = [], 0
    for f in files:
        for text in pq.read_table(f, columns=["text"]).column("text").to_pylist():
            pieces.append(text)
            total += len(text)
            if total >= chars:
                return "\n".join(pieces)[:chars]
    if not pieces:
        raise FileNotFoundError(f"{path} 里没有能读的文字")
    return "\n".join(pieces)[:chars]


def overfit(text: str, out: Path, preset: str = "small", model: list[str] | None = None, steps: int = 300, seq_len: int = 256,
            batch_size: int = 8, prompt_tokens: int = 32, device: str = "auto", tokenizer: Path = DEFAULT_MODEL, lr_mult: float = 1.0) -> dict:
    tok = Tokenizer(tokenizer)
    ids = np.array(tok.encode(text), dtype=np.int64)
    if len(ids) <= seq_len + 1:
        raise ValueError(f"文字太短：只有 {len(ids)} 个词，至少要比 --seq-len（{seq_len}）多一些")
    data = Path(out) / "tokens"
    for split in ("train", "val"):  # 验证集就是训练集：量的是背下来没有
        writer = ShardWriter(data, split, "uint16")
        writer.write(ids)
        writer.close()
    write_meta(data, tok.vocab_size, "uint16", tok.eos_id, "sentencepiece", note="过拟合检查用")
    result = train_pretrain.main([
        "--data", str(data), "--out", str(out), "--preset", preset, "--seq-len", str(seq_len), "--batch-size", str(batch_size),
        "--steps", str(steps), "--warmup-steps", str(min(20, steps // 10)), "--final-lr-frac", "0.1", "--lr-mult", str(lr_mult),
        "--device", device, "--no-compile", "--log-every", str(max(1, steps // 60)), "--eval-every", str(max(1, steps // 3)),
        "--eval-batches", "4", "--save-every", str(steps)] + (["--set", *model] if model else []))

    dev = pick_device(device)
    net = load_model(Path(out) / "ckpt.pt", dev)
    target = ids[prompt_tokens: prompt_tokens + seq_len]
    prompt = torch.from_numpy(ids[None, :prompt_tokens]).to(dev)
    produced = generate(net, prompt, len(target), temperature=0)[0, prompt_tokens:].cpu().numpy()
    match = float((produced[: len(target)] == target).mean())
    report = dict(tokens=int(len(ids)), final_loss=result["loss"], val_loss=result["val_loss"], continuation_match=match,
                  prompt=tok.decode(ids[:prompt_tokens].tolist()), original=tok.decode(target.tolist()), generated=tok.decode(produced.tolist()))
    (Path(out) / "overfit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="过拟合检查")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="parquet 文件或目录，取 text 列")
    parser.add_argument("--chars", type=int, default=3000)
    parser.add_argument("--out", default=str(ROOT / "out" / "overfit"))
    parser.add_argument("--preset", choices=list(PRESETS), default="small")
    parser.add_argument("--set", nargs="*", default=[], metavar="名字=值", help="改模型设置，写法同 training.train_pretrain --set")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-mult", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    report = overfit(load_text(Path(args.input), args.chars), Path(args.out), args.preset, args.set, args.steps, args.seq_len,
                     args.batch_size, device=args.device, lr_mult=args.lr_mult)
    print(f"\n{report['tokens']} 个词；最后训练损失 {report['final_loss']:.4f}，验证损失 {report['val_loss']:.4f}；续写对上 {report['continuation_match']:.1%}")
    print("开头：", report["prompt"])
    print("原文：", report["original"][:200])
    print("续写：", report["generated"][:200])
    return report


if __name__ == "__main__":
    main()
