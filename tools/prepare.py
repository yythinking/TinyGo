"""把文本变成训练用的词数据目录。

    tinygo prepare --input "data/mix/out/nano/*.parquet" --out data/tokens/nano

编码有两种引擎，结果一字节不差：
  gigatoken      装了 gigatoken（pip install gigatoken）、.model 旁边有同名的 .json（HF 格式，tinygo tokenizer export-hf
                 导出，仓库里的正式分词器已经带着）时自动用它。Rust + SIMD，parquet 的列直接交给它，不经过 Python 字符串，
                 M4 Max 上整个 nano（3.9 GB 文本）不到 10 秒。开始时会拿第一批的前几百篇和 SentencePiece 对一遍。
  sentencepiece  其余情况：--workers 个进程并行（默认用满所有核），每个进程单线程跑 SentencePiece，主进程按原顺序收回来写盘。
--engine 可以指定其中一种。

--tokenizer 默认是仓库里的正式分词器 tokenizer/tokenizer.model；也可以换成别的 SentencePiece .model 文件，或者 bytes：
把 UTF-8 的每个字节当成一个词（0 是文档结束符，字节 b 记作 b+1，词表 257），只用来在没有分词器时跑通流程。
用 .model 时会把它复制进输出目录，数据目录自己就能说清楚是用哪个分词器切的。

对话数据（SFT）加 --chat：读 parquet 的 messages 列（每行一段对话，[{role, content}, …]），按 inference.template 的格式编码，
打包成长度为 --seq-len + 1 的行：

    tinygo prepare --chat --input data/sft.parquet --out data/tokens/sft-small --seq-len 2048
"""
import argparse
import collections
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# training.dataset 在用到的函数里才导入：它会带进 torch，编码进程用不着，每个进程省一两秒启动和几百 MB 内存

EOS = 0


def encode_bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16) + 1


def decode_bytes(ids) -> str:
    return bytes(int(i) - 1 for i in ids if i != EOS).decode("utf-8", errors="replace")


def load_encoder(tokenizer: str, threads: int = 4):
    """返回 (一次编一批的函数, 词表大小, 结束符编号, 分词器的说明)。"""
    if tokenizer == "bytes":
        return (lambda texts: [encode_bytes(t) for t in texts]), 257, EOS, "bytes"
    if not tokenizer.endswith(".model") or not Path(tokenizer).exists():
        raise FileNotFoundError(f"--tokenizer 要么是 bytes，要么是一个存在的 SentencePiece .model 文件，收到的是 {tokenizer!r}")
    from inference.tokenizer import Tokenizer

    tok = Tokenizer(tokenizer)
    return (lambda texts: tok.encode(texts, num_threads=threads)), tok.vocab_size, tok.eos_id, "sentencepiece"


def default_tokenizer() -> str:
    from inference.tokenizer import DEFAULT_MODEL

    return str(DEFAULT_MODEL)


_worker = {}


def _start_worker(tokenizer: str) -> None:
    encode, _, eos, _ = load_encoder(tokenizer, threads=1)
    _worker.update(encode=encode, eos=eos)


def _encode_batch(job: tuple) -> dict:
    """一批文档 → {"train": 数组, "val": 数组}：每篇后面接结束符，按篇的全局编号分到训练集或验证集。"""
    first, texts, val_every, dtype = job
    parts = {"train": [], "val": []}
    for k, ids in enumerate(_worker["encode"](texts)):
        parts["val" if val_every and (first + k) % val_every == val_every - 1 else "train"].append(ids)
    out = {}
    for split, docs in parts.items():
        arr, pos = np.empty(sum(len(d) + 1 for d in docs), dtype=dtype), 0
        for d in docs:
            arr[pos:pos + len(d)] = d
            arr[pos + len(d)] = _worker["eos"]
            pos += len(d) + 1
        out[split] = arr
    return out


def load_fast_encoder(tokenizer: str, required: bool = False):
    """gigatoken 编码器：pyarrow 的字符串列 → (每篇的词数, 所有篇的词接在一起)。
    没装 gigatoken、或 .model 旁边没有同名 .json 时返回 None；required 时改成报错，说明缺什么。"""
    json_path = Path(tokenizer).with_suffix(".json")
    try:
        import awkward as ak
        import gigatoken as gt
        import pyarrow as pa
    except ImportError:
        if required:
            raise RuntimeError("要用 gigatoken 引擎得先装：pip install gigatoken") from None
        return None
    if not json_path.exists():
        if required:
            raise RuntimeError(f"gigatoken 要一个和 {Path(tokenizer).name} 逐词一致的 HF 格式文件 {json_path.name}，"
                               f"先跑：python -m tools.tokenizer export-hf {tokenizer}")
        return None
    fast = gt.Tokenizer.from_json(json_path.read_text(encoding="utf-8"))

    def encode(column):
        docs = ak.from_arrow(column.cast(pa.large_string()))
        if isinstance(docs.type.content, ak.types.OptionType):  # 有空值的列：空值当成空文档
            docs = ak.fill_none(docs, "")
        ids = fast.encode_batch(docs)
        return ak.to_numpy(ak.num(ids)).astype(np.int64), ak.to_numpy(ak.flatten(ids))
    return encode


def _split_with_eos(lens: np.ndarray, flat: np.ndarray, first: int, val_every: int, eos: int, dtype: str) -> dict:
    """一批文档（每篇的词数 + 接在一起的词）→ {"train": 数组, "val": 数组}，每篇后面接结束符。"""
    index = first + np.arange(lens.size)
    is_val = (index % val_every == val_every - 1) if val_every else np.zeros(lens.size, dtype=bool)
    token_is_val = np.repeat(is_val, lens)
    out = {}
    for split, docs, tokens in (("train", ~is_val, ~token_is_val), ("val", is_val, token_is_val)):
        n = lens[docs]
        arr = np.empty(int(n.sum()) + n.size, dtype=dtype)
        ends = np.cumsum(n + 1) - 1
        body = np.ones(arr.size, dtype=bool)
        body[ends] = False
        arr[body] = flat[tokens]
        arr[ends] = eos
        out[split] = arr
    return out


def prepare(inputs: list[str], out: str, tokenizer: str | None = None, val_every: int = 1000, shard_tokens: int = 100_000_000,
            workers: int | None = None, batch_docs: int | None = None, engine: str = "auto") -> dict:
    """读 parquet 文件的 text 列，每篇后面接一个结束符。每 val_every 篇里拿一篇放进验证集。返回各部分的词数。
    engine：auto、gigatoken、sentencepiece（见文件开头）。workers 是 sentencepiece 引擎的进程数，默认等于 CPU 核数；1 就在本进程里编。"""
    from training.dataset import ShardWriter, write_meta

    tokenizer = tokenizer or default_tokenizer()
    _, vocab_size, eos, kind = load_encoder(tokenizer)
    if engine == "gigatoken" and kind != "sentencepiece":
        raise RuntimeError("gigatoken 引擎只用于 SentencePiece 分词器")
    fast = load_fast_encoder(tokenizer, required=engine == "gigatoken") if engine != "sentencepiece" and kind == "sentencepiece" else None
    print(f"编码引擎：{'gigatoken' if fast else kind}", flush=True)
    batch_docs = batch_docs or (16384 if fast else 1000)
    dtype = "uint16" if vocab_size <= 1 << 16 else "uint32"
    files = sorted(f for pattern in inputs for f in glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有找到输入文件：{inputs}")
    out = Path(out)
    writers = {"train": ShardWriter(out, "train", dtype, shard_tokens), "val": ShardWriter(out, "val", dtype, shard_tokens)}
    workers = workers or os.cpu_count() or 1

    def jobs():
        first = 0
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_docs, columns=["text"]):
                texts = batch.column("text").to_pylist()
                yield first, texts, val_every, dtype
                first += len(texts)

    def write(result: dict) -> None:
        for split, arr in result.items():
            if arr.size:
                writers[split].write(arr)

    n_docs = 0
    if fast:
        from inference.tokenizer import Tokenizer

        check = Tokenizer(tokenizer)
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_docs, columns=["text"]):
                column = batch.column("text")
                lens, flat = fast(column)
                if n_docs == 0:  # 开始时和 SentencePiece 对一遍，万一不一致就停下，不写出和推理时对不上的数据
                    k = min(256, lens.size)
                    expected = check.encode(column.slice(0, k).to_pylist(), num_threads=4)
                    got = np.split(flat[:int(lens[:k].sum())], np.cumsum(lens[:k])[:-1]) if k else []
                    if any(list(g) != e for g, e in zip(got, expected)):
                        raise RuntimeError("gigatoken 和 SentencePiece 的结果不一致，换 --engine sentencepiece，并到 GitHub 报告")
                write(_split_with_eos(lens, flat, n_docs, val_every, eos, dtype))
                n_docs += lens.size
    elif workers == 1:
        _start_worker(tokenizer)
        for job in jobs():
            write(_encode_batch(job))
            n_docs += len(job[1])
    else:
        # 同时在路上的批数有上限：读得比编得快时不会把整个文件读进内存
        with ProcessPoolExecutor(workers, initializer=_start_worker, initargs=(tokenizer,)) as pool:
            pending = collections.deque()
            for job in jobs():
                pending.append(pool.submit(_encode_batch, job))
                n_docs += len(job[1])
                if len(pending) >= 2 * workers:
                    write(pending.popleft().result())
            while pending:
                write(pending.popleft().result())
    for w in writers.values():
        w.close()
    extra = {}
    if kind == "sentencepiece":
        shutil.copyfile(tokenizer, out / "tokenizer.model")
        extra["tokenizer_file"] = "tokenizer.model"
    write_meta(out, vocab_size, dtype, eos, kind, docs=n_docs, **extra)
    return {"docs": n_docs, "train": writers["train"].total, "val": writers["val"].total}


def ensure_tokens(path: str, chat: bool = False, seq_len: int = 2048, tokenizer: str | None = None) -> str:
    """训练脚本的 --data 可以直接给 parquet：第一次用时切好，放在它旁边的 tokens/ 目录里（对话数据的目录名带上片段长度），
    以后直接用切好的。给的本来就是切好的目录，原样返回。meta.json 最后才写，切到一半中断了，下次会重切。"""
    p = Path(path)
    if p.suffix != ".parquet":
        return path
    if not p.exists():
        raise FileNotFoundError(f"没有找到 {path}。数据集在 https://www.modelscope.cn/datasets/caiwangzsnm/tinybrain-dataset ，"
                                f"下载后用 --data 指过去；也可以用 tinygo data 自己做")
    out = p.parent / "tokens" / (f"{p.stem}-{seq_len}" if chat else p.stem)
    if (out / "meta.json").exists():
        print(f"用已经切好的 {out}", flush=True)
        return str(out)
    print(f"第一次用 {p.name}，先切词到 {out}（只做一次）", flush=True)
    if chat:
        prepare_chat([str(p)], str(out), seq_len, tokenizer)
    else:
        prepare([str(p)], str(out), tokenizer)
    return str(out)


def prepare_chat(inputs: list[str], out: str, seq_len: int, tokenizer: str | None = None, val_every: int = 1000,
                 train_on: str = "assistant", chunk: int = 20_000) -> dict:
    """对话数据：编码、按 seq_len + 1 打包，写出 train/val 的 .ids.bin 和 .mask.bin。返回各部分的统计。"""
    from inference.template import ChatFormat
    from inference.tokenizer import Tokenizer
    from training.dataset import pack, write_meta

    path = tokenizer or default_tokenizer()
    tok = Tokenizer(path)
    fmt = ChatFormat(tok)
    files = sorted(f for pattern in inputs for f in glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有找到输入文件：{inputs}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = "uint16" if tok.vocab_size <= 1 << 16 else "uint32"
    handles = {s: (open(out / f"{s}.ids.bin", "wb"), open(out / f"{s}.mask.bin", "wb")) for s in ("train", "val")}
    pending = {"train": [], "val": []}
    stats = {s: dict(examples=0, dropped_too_long=0, tokens=0, target_tokens=0, rows=0) for s in ("train", "val")}
    n = 0

    def flush(split):
        ids, mask, st = pack(pending[split], seq_len + 1)
        handles[split][0].write(ids.astype(dtype).tobytes())
        handles[split][1].write(mask.tobytes())
        for k in stats[split]:
            stats[split][k] += st[k]
        pending[split] = []

    for path_ in files:
        for batch in pq.ParquetFile(path_).iter_batches(batch_size=1000, columns=["messages"]):
            for messages in batch.column("messages").to_pylist():
                split = "val" if val_every and n % val_every == val_every - 1 else "train"
                pending[split].append(fmt.encode(messages, train_on))
                n += 1
                if len(pending[split]) >= chunk:
                    flush(split)
    for split in pending:
        if pending[split]:
            flush(split)
        for h in handles[split]:
            h.close()
    shutil.copyfile(path, out / "tokenizer.model")
    write_meta(out, tok.vocab_size, dtype, tok.eos_id, "sentencepiece", format="packed", row_len=seq_len + 1, train_on=train_on,
               tokenizer_file="tokenizer.model", stats=stats)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="把文本准备成训练用的词数据")
    parser.add_argument("--input", nargs="+", required=True, help="parquet 文件，可以写通配符")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default=None, help="SentencePiece 的 .model 文件，或者 bytes；默认是 tokenizer/tokenizer.model")
    parser.add_argument("--val-every", type=int, default=1000, help="每多少篇里拿一篇放进验证集；0 表示不留")
    parser.add_argument("--engine", choices=["auto", "gigatoken", "sentencepiece"], default="auto",
                        help="auto：装了 gigatoken 就用它，否则用 SentencePiece；两者结果一字节不差")
    parser.add_argument("--workers", type=int, default=None, help="sentencepiece 引擎用几个进程，默认等于 CPU 核数")
    parser.add_argument("--chat", action="store_true", help="输入是对话（messages 列），打包成 SFT 用的数据")
    parser.add_argument("--seq-len", type=int, default=2048, help="--chat 时每行的片段长度")
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant", help="--chat 时哪些词算损失")
    args = parser.parse_args(argv)
    if args.chat:
        stats = prepare_chat(args.input, args.out, args.seq_len, args.tokenizer, args.val_every, args.train_on)
        for split, st in stats.items():
            print(f"{split}：{st['examples']} 段对话打成 {st['rows']} 行，要学的词 {st['target_tokens'] / 1e6:.2f}M，"
                  f"太长丢掉 {st['dropped_too_long']} 段")
        return
    stats = prepare(args.input, args.out, args.tokenizer, args.val_every, workers=args.workers, engine=args.engine)
    print(f"{stats['docs']} 篇；训练 {stats['train'] / 1e6:.1f}M 词，验证 {stats['val'] / 1e6:.2f}M 词 → {args.out}")


if __name__ == "__main__":
    main()
