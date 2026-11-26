"""标准基准：C-Eval（验证集，52 科）、CMMLU（测试集，67 科）、GSM8K（测试集）。

    tinygo benchmarks prepare                                    # 下载，转成 data/benchmarks/*.jsonl
    tinygo benchmarks run out/small/sft/ckpt.pt --bench ceval cmmlu gsm8k --chat --out out/small/sft/bench.json

选择题：比较模型在「答案：」后面写 A、B、C、D 的概率，取最大的那个。不用生成，所以基座模型也能测。
GSM8K：默认温度 0.8、top-k 50 采样生成解答，取最终答案和标准答案比，判分和数学 RLVR 是同一套（training/rewards/math.py）。
--chat：题目放进对话的 user 轮次（SFT 之后的模型用）；不加时直接接在文本后面（基座模型用）。
--shots k：题目前面先放 k 道例题。选择题用同一科目的开发集，GSM8K 用训练集。

许可：GSM8K 是 MIT；C-Eval 是 CC BY-NC-SA 4.0，CMMLU 是 CC BY-NC 4.0。这几个都只下载到本地做评测，不随仓库或数据集发布。
"""
import argparse
import csv
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import torch

from inference.generate import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, generate, prompt_ids, sampling_generator
from inference.template import ChatFormat
from inference.tokenizer import DEFAULT_MODEL, Tokenizer
from training.trainer_utils import load_model, pick_device
from training.rewards.math import extract_answer, same_answer
from training.rollout import sample_group

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "data" / "benchmarks"
URLS = {
    "ceval-exam.zip": "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/c-eval/ceval-exam.zip",
    "cmmlu.zip": "https://www.modelscope.cn/api/v1/datasets/modelscope/cmmlu/repo?Revision=master&FilePath=cmmlu_v1_0_1.zip",
    "gsm8k_train.jsonl": "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/gsm8k/train.jsonl",
    "gsm8k_test.jsonl": "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/gsm8k/test.jsonl",
}
LETTERS = "ABCD"
# 选择题从哪个压缩包的哪一份取：（评测用的那份，例题用的那份）
CHOICE = {"ceval": ("ceval-exam.zip", "val", "dev"), "cmmlu": ("cmmlu.zip", "test", "dev")}
GSM8K_HINT = "\n请一步一步算，最后一行写「答案：」加上最终的数。"


def read_choices(zip_path: Path, split: str) -> list[dict]:
    """压缩包里 split/ 下的每个 csv 是一个科目。C-Eval 的列是 id,question,A,B,C,D,answer；CMMLU 是 ,Question,A,B,C,D,Answer。"""
    items = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.startswith(split + "/") and n.endswith(".csv")):
            subject = re.sub(rf"_{split}$", "", Path(name).stem)
            with z.open(name) as f:
                for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                    row = {k.lower(): (v or "").strip() for k, v in row.items() if k}
                    if row.get("answer") in tuple(LETTERS):
                        items.append(dict(subject=subject, question=row["question"], choices=[row[c.lower()] for c in LETTERS], answer=row["answer"]))
    return items


def read_gsm8k(path: Path) -> list[dict]:
    """标准答案在解答的「#### 」后面；解答里的 <<算式>> 是给计算器的标注，去掉。"""
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            solution, _, answer = row["answer"].rpartition("####")
            out.append(dict(question=row["question"].strip(), solution=re.sub(r"<<[^>]*>>", "", solution).strip(), answer=answer.strip().replace(",", "")))
    return out


def prepare(out_dir: Path = BENCH_DIR) -> dict:
    """下载（已经有的不重下）并转成 jsonl。返回每个文件的条数。"""
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if not (raw / name).exists():
            urllib.request.urlretrieve(url, raw / (name + ".part"))
            (raw / (name + ".part")).rename(raw / name)
    files = {f"{b}_{part}": read_choices(raw / z, split) for b, (z, test, dev) in CHOICE.items() for part, split in (("test", test), ("dev", dev))}
    files.update(gsm8k_test=read_gsm8k(raw / "gsm8k_test.jsonl"), gsm8k_dev=read_gsm8k(raw / "gsm8k_train.jsonl"))
    for name, rows in files.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return {name: len(rows) for name, rows in files.items()}


def load(bench: str, part: str = "test", bench_dir: Path = BENCH_DIR) -> list[dict]:
    path = Path(bench_dir) / f"{bench}_{part}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"没有 {path}，先跑 python -m eval.benchmarks prepare")
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _choice_block(item: dict, with_answer: bool) -> str:
    lines = [item["question"]] + [f"{c}. {t}" for c, t in zip(LETTERS, item["choices"])]
    return "\n".join(lines) + "\n答案：" + (item["answer"] + "\n\n" if with_answer else "")


def choice_prompt(item: dict, shots: list[dict], fmt: ChatFormat, chat: bool) -> list[int]:
    """模型接着要写的就是选项字母。"""
    head = f"以下是关于「{item['subject']}」的单项选择题，请直接给出正确答案的选项。\n\n"
    body = head + "".join(_choice_block(s, True) for s in shots) + _choice_block(item, False)
    if not chat:
        return fmt.tok.encode(body)
    question, _, _ = body.rpartition("答案：")
    return fmt.prompt([{"role": "user", "content": question.rstrip()}]) + fmt.tok.encode("答案：")


@torch.no_grad()
def choice_accuracy(model, fmt: ChatFormat, items: list[dict], dev: list[dict], shots: int = 0, chat: bool = False) -> dict:
    """每道题算一次前向，看最后一个位置上四个字母哪个概率最大。放不下的题先少放例题，还放不下就跳过。"""
    device = next(model.parameters()).device
    letter_ids = [fmt.tok.encode(c) for c in LETTERS]
    if any(len(i) != 1 for i in letter_ids):
        raise ValueError(f"分词器把选项字母切成了多个词：{letter_ids}")
    letter_ids = torch.tensor([i[0] for i in letter_ids], device=device)
    by_subject_dev = {}
    for d in dev:
        by_subject_dev.setdefault(d["subject"], []).append(d)
    model.eval()
    right, skipped, per_subject = 0, 0, {}
    for item in items:
        examples = by_subject_dev.get(item["subject"], [])[:shots]
        ids = choice_prompt(item, examples, fmt, chat)
        while len(ids) > model.cfg.max_seq_len and examples:
            examples = examples[:-1]
            ids = choice_prompt(item, examples, fmt, chat)
        if len(ids) > model.cfg.max_seq_len:
            skipped += 1
            continue
        logits = model(torch.tensor([ids], device=device)).logits[0, -1]
        ok = LETTERS[int(logits[letter_ids].argmax())] == item["answer"]
        right += ok
        s = per_subject.setdefault(item["subject"], [0, 0])
        s[0] += ok
        s[1] += 1
    n = len(items) - skipped
    return dict(n=n, skipped=skipped, acc=round(right / max(1, n), 4),
                by_subject={k: round(r / t, 4) for k, (r, t) in sorted(per_subject.items())})


def _gsm8k_shots(shots: list[dict]) -> str:
    return "".join(f"问题：{s['question']}\n解答：{s['solution']}\n答案：{s['answer']}\n\n" for s in shots)


@torch.no_grad()
def gsm8k_accuracy(model, fmt: ChatFormat, items: list[dict], dev: list[dict], shots: int = 0, chat: bool = False, max_new: int = 384,
                   temperature: float = DEFAULT_TEMPERATURE, top_k: int | None = DEFAULT_TOP_K, seed: int = 0) -> dict:
    device = next(model.parameters()).device
    generator = sampling_generator(device, seed)
    right, skipped = 0, 0
    for item in items:
        if chat:
            messages = [{"role": "user", "content": _gsm8k_shots(dev[:shots]) + item["question"] + GSM8K_HINT}]
            if len(fmt.prompt(messages)) + 16 > model.cfg.max_seq_len:
                skipped += 1
                continue
            text = sample_group(model, fmt, messages, 1, max_new, temperature=temperature, top_k=top_k, generator=generator)[0].text
        else:
            prompt = prompt_ids(fmt.tok, _gsm8k_shots(dev[:shots]) + f"问题：{item['question']}\n解答：")
            if len(prompt) + 16 > model.cfg.max_seq_len:
                skipped += 1
                continue
            out = generate(model, torch.tensor([prompt], device=device), max_new, temperature=temperature,
                           top_k=top_k, eos_id=fmt.tok.eos_id, generator=generator)
            text = fmt.tok.decode(out[0, len(prompt):].tolist()).split("\n\n问题：")[0]
        right += same_answer(extract_answer(text), item["answer"])
    n = len(items) - skipped
    return dict(n=n, skipped=skipped, acc=round(right / max(1, n), 4))


def run_benchmarks(model, fmt: ChatFormat, benches=("ceval", "cmmlu", "gsm8k"), limit: int | None = None, shots: int = 0, chat: bool = False,
                   bench_dir: Path = BENCH_DIR, max_new: int = 384,
                   temperature: float = DEFAULT_TEMPERATURE, top_k: int | None = DEFAULT_TOP_K, seed: int = 0) -> dict:
    """limit：每个基准最多评前多少道（按科目均匀取，免得只测到排在前面的几科）。"""
    report = {}
    for bench in benches:
        items, dev = load(bench, "test", bench_dir), load(bench, "dev", bench_dir)
        if limit and len(items) > limit:
            step = len(items) / limit
            items = [items[int(i * step)] for i in range(limit)]
        if bench == "gsm8k":
            report[bench] = gsm8k_accuracy(model, fmt, items, dev, shots, chat, max_new, temperature, top_k, seed)
        else:
            report[bench] = choice_accuracy(model, fmt, items, dev, shots, chat)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="标准基准：C-Eval、CMMLU、GSM8K")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("ckpt")
    r.add_argument("--bench", nargs="+", default=["ceval", "cmmlu", "gsm8k"], choices=["ceval", "cmmlu", "gsm8k"])
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--shots", type=int, default=0)
    r.add_argument("--chat", action="store_true")
    r.add_argument("--max-new", type=int, default=384)
    r.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    r.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="auto")
    r.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False))
        return
    device = pick_device(args.device)
    model = load_model(Path(args.ckpt), device).eval()
    report = run_benchmarks(model, ChatFormat(Tokenizer(DEFAULT_MODEL)), args.bench, args.limit, args.shots, args.chat, max_new=args.max_new,
                            temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    for bench, r in report.items():
        print(f"{bench:<6} 准确率 {r['acc']:.2%}（{r['n']} 道，跳过 {r['skipped']} 道）")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(ckpt=args.ckpt, chat=args.chat, shots=args.shots,
            sampling=dict(temperature=args.temperature, top_k=args.top_k, seed=args.seed), **report), ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
