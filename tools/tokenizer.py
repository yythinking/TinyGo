"""分词器：从各个来源按配比抽一份样本，训练 SentencePiece（BPE），再在留出来的文档上量压缩率。

    tinygo tokenizer sample --tokens 40000000        # 抽样本 → out/tokenizer/corpus.txt、heldout.jsonl
    tinygo tokenizer train --vocab 32768              # 训练 → out/tokenizer/sp32768.model
    tinygo tokenizer eval out/tokenizer/sp32768.model # 各类文本上一个词平均几个字
    tinygo tokenizer export-hf tokenizer/tokenizer.model  # 导出逐词一致的 HF 格式 → tokenizer/tokenizer.json

样本直接从原始文件现洗（和 dataset/pretrain.py 同一套清洗），所以还没整批洗完的来源也能用。
每个来源只读少数几个分片（默认 3 个，在文件列表里均匀地取，大来源的不同子来源都能碰到），每个分片读开头的一部分。
配比按估算的 token 数算，和 dataset/mix.py 的 MIX 一致；还没下载的来源跳过，份额在同类里补上。

训练的设置：
  - 不做任何归一化、不加前导空格、不合并连续空格：解码回来和原文一字不差（代码缩进、全角标点都保住）。
  - 字节兜底（byte_fallback）：词表外的字拆成 UTF-8 字节，永远不会出现未知词；换行也是这样编码的。字符覆盖率 0.99995。
  - 数字拆成单个数位（对算术有好处）。
  - 0 未知、1 开头、2 结束（文档之间的分隔）、3 填充；再留 8 个控制符给以后的对话格式。
"""
import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm

from dataset import pretrain
from dataset.clean import clean_with_reason, est_tokens
from dataset.mix import MIX, allocate
from dataset.sources import PRETRAIN, ROOT

from inference.tokenizer import BOS, CONTROL, DEFAULT_MODEL, EOS, PAD, UNK, VOCAB_SIZE, Tokenizer  # noqa: F401  正式分词器在推理包里

OUT = ROOT / "out" / "tokenizer"   # 训练分词器用的样本（不进仓库）
MAX_LINE = 4000  # 训练样本里一行最多多少字，更长的切开（SentencePiece 默认跳过太长的句子）
# 字符覆盖率：出现频率排在最后这么一点的字不进词表、只能拆成字节。0.9995 时「刨、榈、栾、卞、锑」这类常用字都被排除了，所以取得更高。
COVERAGE = 0.99995


def available_sources() -> set[str]:
    """已经下载好、至少有一个能读的原始文件的来源。"""
    found = set()
    for name, spec in PRETRAIN.items():
        if (pretrain.RAW / spec.get("raw", name)).is_dir() and pretrain.raw_files(name):
            found.add(name)
    return found


def source_budgets(total: float, available: set[str]) -> dict[str, float]:
    """按 MIX 的配比把 total 个 token 分给各个来源；没有的来源分到 0，它的份额先在同类里补，同类都没有再分给别的类。"""
    big = float("inf")
    cats = allocate(total, {c: big if available & set(srcs) else 0 for c, (_, srcs) in MIX.items()}, {c: share for c, (share, _) in MIX.items()})
    budgets = {}
    for cat, (_, srcs) in MIX.items():
        budgets.update(allocate(cats[cat], {s: big if s in available else 0 for s in srcs}, srcs))
    return budgets


def pick_shards(files: list, n: int) -> list:
    """在文件列表里均匀地取 n 个（包括第一个和最后一个）。"""
    if len(files) <= n:
        return list(files)
    if n == 1:
        return [files[0]]
    return [files[round(i * (len(files) - 1) / (n - 1))] for i in range(n)]


def iter_clean_docs(name: str, path: Path, max_rows: int):
    """一个原始文件里洗好的文档（只读前 max_rows 行）。"""
    spec = PRETRAIN[name]
    rows = pretrain.iter_rows(path)
    rows = spec["regroup"](rows) if spec.get("regroup") else rows
    for i, row in enumerate(rows):
        if i >= max_rows:
            return
        raws = spec["text"](row)
        score = spec["score"](row) if spec.get("score") else None
        for raw in [raws] if isinstance(raws, str) else raws:
            doc, _ = clean_with_reason(raw, spec["kind"], native_score=score, markdown=spec.get("markdown", False))
            if doc is not None:
                yield doc["text"]


def training_lines(text: str):
    """一篇文档拆成训练用的行：去掉空行，太长的行切开。"""
    for line in text.split("\n"):
        if line.strip():
            for start in range(0, len(line), MAX_LINE):
                yield line[start:start + MAX_LINE]


def sample_corpus(total_tokens: float, out_dir: Path = OUT, shards: int = 3, holdout_every: int = 50, max_rows: int = 50_000) -> dict:
    """按配比抽样本：训练用的行写进 corpus.txt，每 holdout_every 篇留一篇写进 heldout.jsonl。返回每个来源的统计。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = source_budgets(total_tokens, available_sources())
    stats = {}
    with open(out_dir / "corpus.txt", "w", encoding="utf-8") as corpus, open(out_dir / "heldout.jsonl", "w", encoding="utf-8") as heldout:
        for name, budget in budgets.items():
            if budget <= 0:
                continue
            category = next(c for c, (_, srcs) in MIX.items() if name in srcs)
            files = pick_shards(pretrain.raw_files(name), shards)
            got = dict(category=category, budget=int(budget), tokens=0, chars=0, docs=0, heldout=0, shards=[f.name for f in files])
            for path in files:
                target = got["tokens"] + budget / len(files)
                for text in iter_clean_docs(name, path, max_rows):
                    if got["tokens"] >= target:
                        break
                    got["docs"] += 1
                    if holdout_every and got["docs"] % holdout_every == 0:
                        heldout.write(json.dumps(dict(source=name, category=category, text=text), ensure_ascii=False) + "\n")
                        got["heldout"] += 1
                        continue
                    for line in training_lines(text):
                        corpus.write(line + "\n")
                    got["tokens"] += est_tokens(text)
                    got["chars"] += len(text)
            stats[name] = got
            print(f"  {name:<18}{category:<8} 要 {budget / 1e6:6.2f}M，抽到 {got['tokens'] / 1e6:6.2f}M token（{got['docs']} 篇，{len(files)} 个分片）", flush=True)
    (out_dir / "_sample.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def train(vocab_size: int, corpus: Path = OUT / "corpus.txt", out_dir: Path = OUT, threads: int = 4, sentences: int = 6_000_000,
          coverage: float = COVERAGE, **extra) -> Path:
    """训练一个 BPE 分词器，返回 .model 文件的路径。sentences 是最多读多少行（随机抽）。"""
    prefix = Path(out_dir) / f"sp{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size, model_type="bpe",
        character_coverage=coverage, byte_fallback=True, split_digits=True,
        normalization_rule_name="identity", add_dummy_prefix=False, remove_extra_whitespaces=False, allow_whitespace_only_pieces=True,
        unk_id=UNK, bos_id=BOS, eos_id=EOS, pad_id=PAD, control_symbols=CONTROL,
        input_sentence_size=sentences, shuffle_input_sentence=True, max_sentence_length=4 * MAX_LINE * 2,
        num_threads=threads, minloglevel=2, **extra)
    strip_paths(prefix.with_suffix(".model"))
    return prefix.with_suffix(".model")


def strip_paths(model_path: Path) -> None:
    """SentencePiece 把训练语料和输出前缀的绝对路径记在模型文件里（trainer_spec），对切词没有影响。
    只留文件名，免得发布模型时带出本机的目录。"""
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(Path(model_path).read_bytes())
    names = [Path(p).name for p in m.trainer_spec.input]
    del m.trainer_spec.input[:]
    m.trainer_spec.input.extend(names)
    m.trainer_spec.model_prefix = Path(m.trainer_spec.model_prefix).name
    Path(model_path).write_bytes(m.SerializeToString())


def export_hf(model_path: Path, out: Path | None = None) -> Path:
    """导出一个和 .model 逐词一致的 HF 格式分词器（默认写到同名 .json），tools.prepare 用它交给 gigatoken 编码，快一个数量级。

    做法和 Llama 2 的官方转换一样：空格换成 ▁，BPE 按「合并结果在词表里的编号」排合并顺序（SentencePiece 按分数挑，
    分数和编号同序），字节兜底。不需要预分词：训练时已经按空白、文字种类和数字切过，词表里没有跨这些边界的词。
    控制符、未知词和字节词不当作合并的结果：SentencePiece 从来不会从正文里拼出它们。要装 tokenizers。"""
    from tokenizers import Tokenizer as HFTokenizer, models, normalizers

    sp = Tokenizer(model_path).sp
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    special = {i for i in range(sp.get_piece_size()) if sp.is_control(i) or sp.is_unknown(i) or sp.is_byte(i)}
    merges = sorted(((i, k), (piece[:k], piece[k:])) for piece, i in vocab.items() if i not in special
                    for k in range(1, len(piece)) if piece[:k] in vocab and piece[k:] in vocab)
    hf = HFTokenizer(models.BPE(vocab=vocab, merges=[m for _, m in merges], byte_fallback=True, unk_token=sp.id_to_piece(UNK), fuse_unk=True))
    hf.normalizer = normalizers.Replace(" ", "▁")
    out = Path(out) if out else Path(model_path).with_suffix(".json")
    hf.save(str(out))
    return out


def evaluate(model_path: Path, heldout: Path = OUT / "heldout.jsonl") -> dict:
    """在留出来的文档上，按类别量：一个词平均几个字、字节兜底的词占多少、和流水线估算的 token 数差多少；并检查解码是否和原文一样。"""
    tok = Tokenizer(model_path)
    byte_ids = {i for i in range(tok.vocab_size) if tok.sp.is_byte(i)}
    docs = [json.loads(line) for line in open(heldout, encoding="utf-8")]
    ids = tok.encode([d["text"] for d in docs], num_threads=4)
    by_cat = {}
    for d, seq in zip(docs, ids):
        s = by_cat.setdefault(d["category"], dict(docs=0, chars=0, tokens=0, bytes=0, est=0, mismatched=0))
        s["docs"] += 1
        s["chars"] += len(d["text"])
        s["tokens"] += len(seq)
        s["bytes"] += sum(i in byte_ids for i in seq)
        s["est"] += est_tokens(d["text"])
        s["mismatched"] += tok.decode(seq) != d["text"]
    total = {k: sum(s[k] for s in by_cat.values()) for k in ("docs", "chars", "tokens", "bytes", "est", "mismatched")}
    result = {}
    for cat, s in [*sorted(by_cat.items(), key=lambda kv: -kv[1]["tokens"]), ("合计", total)]:
        result[cat] = dict(docs=s["docs"], chars_per_token=s["chars"] / max(1, s["tokens"]), byte_share=s["bytes"] / max(1, s["tokens"]),
                           tokens_vs_estimate=s["tokens"] / max(1, s["est"]), mismatched=s["mismatched"])
    return result


def print_eval(name: str, result: dict) -> None:
    print(f"== {name}")
    print(f"  {'类别':<10}{'篇数':>6}{'字/词':>8}{'字节兜底':>10}{'实际/估算':>10}{'解码不一致':>10}")
    for cat, r in result.items():
        print(f"  {cat:<10}{r['docs']:>6}{r['chars_per_token']:>8.2f}{r['byte_share']:>10.2%}{r['tokens_vs_estimate']:>10.2f}{r['mismatched']:>10}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="抽样本、训练、评测分词器")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--tokens", type=float, default=40e6, help="按估算一共抽多少 token")
    p.add_argument("--shards", type=int, default=3, help="每个来源读几个分片")
    p.add_argument("--out", default=str(OUT))
    p = sub.add_parser("train")
    p.add_argument("--vocab", type=int, nargs="+", default=[VOCAB_SIZE], help="可以给几个，挨个训练")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=str(OUT))
    p = sub.add_parser("eval")
    p.add_argument("models", nargs="+")
    p.add_argument("--heldout", default=str(OUT / "heldout.jsonl"))
    p = sub.add_parser("export-hf", help="导出逐词一致的 HF 格式（同名 .json）")
    p.add_argument("model", nargs="?", default=str(DEFAULT_MODEL))
    args = parser.parse_args(argv)
    if args.command == "export-hf":
        print("写到了", export_hf(Path(args.model)))
        return
    if args.command == "sample":
        stats = sample_corpus(args.tokens, Path(args.out), args.shards)
        print(f"共抽到 {sum(s['tokens'] for s in stats.values()) / 1e6:.1f}M token、{sum(s['chars'] for s in stats.values()) / 1e6:.1f}M 字，"
              f"留出 {sum(s['heldout'] for s in stats.values())} 篇做评测")
    elif args.command == "train":
        for vocab in args.vocab:
            path = train(vocab, Path(args.out) / "corpus.txt", Path(args.out), args.threads)
            print(f"训练好 {path}", flush=True)
    else:
        for model in args.models:
            print_eval(model, evaluate(Path(model), Path(args.heldout)))


if __name__ == "__main__":
    main()
