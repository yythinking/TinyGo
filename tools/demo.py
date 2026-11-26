"""在一个很小的模型上把整条链跑一遍，确认每个阶段都是通的。不是为了效果，是为了「跑通」。

    tinygo demo --out out/demo --judge-url http://localhost:8080/v1 --judge-model Qwen3.8-27B-4bit --judge-workers 2

阶段（做完的会跳过，可以随时中断再接着跑；--only 只跑某几个阶段）：
  sample     预训练小样本：有 data/pretrain_nano.parquet 时取它的开头一段（文件内部已经打乱，就是一份按配比的样本），
             没有时从洗好的数据里按配比抽；切词
  pretrain   预训练
  anneal     退火：九成预训练数据 + 一成对话格式的数据（整段算损失）
  sft        SFT（small 的语料）
  ── small 的路线 ──
  agent      agent RL（基础工具调用），同时练指令跟随、中文阅读、诗词填空
  rlaif      RLAIF（评审模型打分：通用、写作、诗词）
  ── medium / large 的路线（这里用同一个小模型代替）──
  experts    数学、代码、文学三个专家，各自在领域数据上接着 SFT
  rlvr       数学专家做 RLVR（数学题对答案、逻辑题）
  rlvr_code  代码专家做 RLVR（跑测试用例）
  mopd       MOPD：学生 = SFT 模型，老师 = 三个专家（通用题的老师是 SFT 模型自己）
  mopd_agent MOPD 之后的 RL：工具调用、科学选择题、指令跟随、中文阅读、诗词填空
  mopd_rlaif 最后的 RLAIF
  eval       给每个阶段的存档做评测，汇总成一张表

没有评审模型（--judge-url 空着）时，rlaif、mopd_rlaif 两步跳过，评测里也不算评审题。
评审模型用 mlx_lm.server 时一次只算一个请求，要加 --judge-workers 1 或 2，否则排队的请求会超时。
"""
import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataset.mix import MIX
from eval import suite
from tools.prepare import prepare, prepare_chat
from training import train_distill, train_pretrain, train_rl, train_sft

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# 演示用的小模型：主干 3.3M，合计约 23M（词表 32768）
DEMO_MODEL = dict(preset="small", dim=256, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=1024, window=256, set=["engram_head_dim=16"])
NANO = DATA / "pretrain_nano.parquet"
SMALL_RL = ("agent", "instruct", "reading", "cloze")            # small 的 agent RL 一步里练的题（都是程序判分）
EXPERT_RL = {"math": ("math", "logic"), "code": ("code",)}      # 两个专家各自的 RLVR
AFTER_MOPD_RL = ("agent", "choice", "instruct", "reading", "cloze")
STAGES = ("sample", "pretrain", "anneal", "sft", "agent", "rlaif", "experts", "rlvr", "rlvr_code", "mopd", "mopd_agent", "mopd_rlaif", "eval")


def nano_sample(out: Path, tokens: float, nano: Path = NANO) -> dict:
    """从正式的 nano 文件开头取够 tokens 个估算 token，写成 out/pretrain_sample.parquet。"""
    texts, got, by_source = [], 0, {}
    for batch in pq.ParquetFile(nano).iter_batches(columns=["text", "source", "est_tokens"], batch_size=4096):
        for text, source, n in zip(*(batch.column(c).to_pylist() for c in ("text", "source", "est_tokens"))):
            if got >= tokens:
                break
            texts.append(text)
            got += n
            by_source[source] = by_source.get(source, 0) + n
        if got >= tokens:
            break
    pq.write_table(pa.table({"text": texts}), out / "pretrain_sample.parquet", compression="zstd")
    return dict(docs=len(texts), est_tokens_by_source=by_source, taken_from=str(nano))


def pretrain_sample(out: Path, tokens: float, seed: int = 0) -> dict:
    """从 data/clean 里已经洗完的来源，按 MIX 的类别比例取文档，写成一个 parquet。没洗完的来源跳过，份额由同类别或别的类别补。"""
    from dataset.clean import est_tokens
    from dataset.mix import REAL_PER_EST, allocate

    clean = DATA / "clean"
    ready = {p.name for p in clean.iterdir() if (p / "_stats.json").exists()}
    big = float("inf")
    cats = allocate(tokens, {c: big if ready & set(s) else 0 for c, (_, s) in MIX.items()}, {c: share for c, (share, _) in MIX.items()})
    texts, taken = [], {}
    for cat, (_, srcs) in MIX.items():
        budget = allocate(cats[cat] / REAL_PER_EST.get(cat, 1.0), {s: big if s in ready else 0 for s in srcs}, srcs)
        for name, want in budget.items():
            got = 0
            for f in sorted((clean / name).glob("*.parquet")):
                if got >= want:
                    break
                t = pq.read_table(f, columns=["text", "est_tokens"])
                for text, n in zip(t.column("text").to_pylist(), t.column("est_tokens").to_pylist()):
                    if got >= want:
                        break
                    texts.append(text)
                    got += n
            if got:
                taken[name] = got
    path = out / "pretrain_sample.parquet"
    pq.write_table(pa.table({"text": texts}), path, compression="zstd")
    return dict(docs=len(texts), est_tokens_by_source=taken)


def argv(**opts) -> list[str]:
    """把 seq_len=512、no_compile=True 这样的写法变成命令行参数 ["--seq-len", "512", "--no-compile"]。每个阶段就是一行命令。"""
    out = []
    for key, value in opts.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            out.append(flag)
        elif isinstance(value, list):
            out += [flag, *map(str, value)]
        elif value not in (None, False, ""):
            out += [flag, str(value)]
    return out


def subset(src: Path, dst: Path, keep) -> Path:
    rows = [r for r in pq.read_table(src).to_pylist() if keep(r)]
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(src)), dst)
    return dst


def run(out: Path, tokens: float = 80e6, judge_url: str = "", judge_model: str = "", only: list[str] | None = None, device: str = "auto",
        seq_len: int = 512, batch_size: int = 32, judge_workers: int = 16) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    want = lambda s: only is None or s in only
    done = lambda s: (out / s / "ckpt.pt").exists()
    log = {}
    t0 = time.time()

    def stage(name):
        print(f"\n===== {name}（已用 {(time.time() - t0) / 60:.0f} 分钟）", flush=True)

    tok = out / "tokens"
    common = dict(device=device, no_compile=True, log_every=50, eval_batches=8)
    ready = lambda name: (tok / name / "meta.json").exists()  # 每份数据各自检查：换了 SFT 语料，删掉对应的目录就会重新准备
    if want("sample"):
        if not ready("pt"):
            stage("sample")
            log["sample"] = nano_sample(out, tokens) if NANO.exists() else pretrain_sample(out, tokens)
            prepare([str(out / "pretrain_sample.parquet")], str(tok / "pt"), val_every=200)
        if not ready("chat_all"):
            prepare_chat([str(DATA / "sft_small.parquet")], str(tok / "chat_all"), seq_len=seq_len, train_on="all", val_every=100)
        if not ready("sft"):
            prepare_chat([str(DATA / "sft_small.parquet")], str(tok / "sft"), seq_len=1024, val_every=100)
        for key in ("math", "code", "literature"):
            if not ready(f"expert_{key}"):
                prepare_chat([str(DATA / f"sft_expert_{key}.parquet")], str(tok / f"expert_{key}"), seq_len=1024, val_every=50)
    if want("pretrain") and not done("pretrain"):
        stage("pretrain")
        log["pretrain"] = train_pretrain.main(argv(data=tok / "pt", out=out / "pretrain", **DEMO_MODEL, seq_len=seq_len, batch_size=batch_size, epochs=1,
                                                   warmup_steps=200, final_lr_frac=0.1, eval_every=500, save_every=10**9, **common))
    if want("anneal") and not done("anneal"):
        stage("anneal")  # 预训练最后的学习率是起步值的 0.1 倍，退火从这里接着降到 0
        log["anneal"] = train_pretrain.main(argv(data=f"{tok / 'pt'}:0.9,{tok / 'chat_all'}:0.1", out=out / "anneal", init=out / "pretrain" / "ckpt.pt",
                                                 seq_len=seq_len, batch_size=batch_size, epochs=0.1, warmup_steps=0, lr_mult=0.1,
                                                 final_lr_frac=0.0, eval_every=500, save_every=10**9, **common))
    sft_common = dict(common, batch_size=16, eval_every=100)
    if want("sft") and not done("sft"):
        stage("sft")
        log["sft"] = train_sft.main(argv(data=tok / "sft", out=out / "sft", init=out / "anneal" / "ckpt.pt", steps=800, warmup_steps=30, save_every=800,
                                         **sft_common))
    rl_common = dict(device=device, max_new=160, prompts_per_step=8, group_size=4, save_every=10**9, log_every=1, judge_url=judge_url,
                     judge_model=judge_model or None, judge_workers=judge_workers)
    rl_small = DATA / "rl_small.parquet"
    if want("agent") and not done("agent"):
        stage("agent")
        tasks = subset(rl_small, out / "tasks_agent.parquet", lambda r: r["reward"] in SMALL_RL)
        log["agent"] = train_rl.main(argv(init=out / "sft" / "ckpt.pt", tasks=tasks, out=out / "agent", steps=30, max_turns=3, **{**rl_common, "judge_url": ""}))
    if want("rlaif") and judge_url and not done("rlaif"):
        stage("rlaif")
        tasks = subset(rl_small, out / "tasks_rlaif.parquet", lambda r: r["reward"] in ("judge", "poetry"))
        log["rlaif"] = train_rl.main(argv(init=out / "agent" / "ckpt.pt", tasks=tasks, out=out / "rlaif", steps=10, **{**rl_common, "prompts_per_step": 4}))
    if want("experts"):
        for key in ("math", "code", "literature"):
            if not done(f"expert_{key}"):
                stage(f"expert_{key}")
                log[f"expert_{key}"] = train_sft.main(argv(data=tok / f"expert_{key}", out=out / f"expert_{key}", init=out / "sft" / "ckpt.pt",
                                                           steps=300, warmup_steps=10, save_every=300, **sft_common))
    if want("rlvr") and not done("rlvr"):
        stage("rlvr")
        tasks = subset(DATA / "rl_large.parquet", out / "tasks_math.parquet", lambda r: r["reward"] in EXPERT_RL["math"])
        log["rlvr"] = train_rl.main(argv(init=out / "expert_math" / "ckpt.pt", tasks=tasks, out=out / "rlvr", steps=20,
                                         **{**rl_common, "max_new": 256, "judge_url": ""}))
    if want("rlvr_code") and not done("rlvr_code"):
        stage("rlvr_code")
        tasks = subset(DATA / "rl_large.parquet", out / "tasks_code.parquet", lambda r: r["reward"] in EXPERT_RL["code"])
        if pq.read_metadata(tasks).num_rows:
            log["rlvr_code"] = train_rl.main(argv(init=out / "expert_code" / "ckpt.pt", tasks=tasks, out=out / "rlvr_code", steps=10,
                                                  **{**rl_common, "max_new": 384, "judge_url": ""}))
    if want("mopd") and not done("mopd"):
        stage("mopd")
        teachers = {"数学": str(out / "rlvr" / "ckpt.pt"), "代码": str(out / ("rlvr_code" if done("rlvr_code") else "expert_code") / "ckpt.pt"),
                    "文学": str(out / "expert_literature" / "ckpt.pt"), "通用": str(out / "sft" / "ckpt.pt")}
        log["mopd"] = train_distill.main(argv(init=out / "sft" / "ckpt.pt", prompts=DATA / "mopd_prompts.parquet", out=out / "mopd",
                                              teachers=[f"{d}={p}" for d, p in teachers.items()], steps=40, prompts_per_step=8, samples_per_prompt=2,
                                              max_new=160, device=device, save_every=40))
    if want("mopd_agent") and not done("mopd_agent"):
        stage("mopd_agent")
        tasks = subset(DATA / "rl_large.parquet", out / "tasks_after_mopd.parquet", lambda r: r["reward"] in AFTER_MOPD_RL)
        log["mopd_agent"] = train_rl.main(argv(init=out / "mopd" / "ckpt.pt", tasks=tasks, out=out / "mopd_agent", steps=30, max_turns=3,
                                               **{**rl_common, "judge_url": ""}))
    if want("mopd_rlaif") and judge_url and not done("mopd_rlaif"):
        stage("mopd_rlaif")
        tasks = out / "tasks_rlaif.parquet"
        if not tasks.exists():
            subset(DATA / "rl_large.parquet", tasks, lambda r: r["reward"] in ("judge", "poetry"))
        log["mopd_rlaif"] = train_rl.main(argv(init=out / ("mopd_agent" if done("mopd_agent") else "mopd") / "ckpt.pt", tasks=tasks, out=out / "mopd_rlaif",
                                               steps=10, **{**rl_common, "prompts_per_step": 4}))
    if want("eval"):
        stage("eval")
        table = {}
        for name in ("pretrain", "anneal", "sft", "agent", "rlaif", "expert_math", "rlvr", "expert_code", "rlvr_code", "mopd", "mopd_agent", "mopd_rlaif"):
            if done(name):
                print(f"  评测 {name}", flush=True)
                table[name] = suite.run(str(out / name / "ckpt.pt"), str(DATA / "eval_tasks.parquet"), n_tasks=30, judge_url=judge_url,
                                        judge_model=judge_model, max_new=200, device=device, out=str(out / name / "eval.json"), judge_workers=judge_workers,
                                        benches=["ceval", "cmmlu", "gsm8k"] if (ROOT / "data" / "benchmarks" / "ceval_test.jsonl").exists() else (),
                                        bench_limit=100)
        (out / "eval_summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        log["eval"] = {k: {kk: v.get(kk) for kk in ("heldout_loss", "tasks", "benchmarks")} for k, v in table.items()}
    (out / "demo_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n全部完成，用时 {(time.time() - t0) / 60:.0f} 分钟", flush=True)
    return log


def main(argv=None):
    parser = argparse.ArgumentParser(description="小模型上把整条链跑一遍")
    parser.add_argument("--out", default=str(ROOT / "out" / "demo"))
    parser.add_argument("--tokens", type=float, default=80e6, help="预训练用多少 token")
    parser.add_argument("--judge-url", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-workers", type=int, default=16, help="同时发几个评审请求；mlx_lm.server 用 1–2")
    parser.add_argument("--only", nargs="*", choices=STAGES)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    return run(Path(args.out), args.tokens, args.judge_url, args.judge_model, args.only, args.device, judge_workers=args.judge_workers)


if __name__ == "__main__":
    main()
