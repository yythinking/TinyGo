"""从头到尾做出全部数据：按顺序跑各步，每一步开跑前先检查它要的输入在不在，缺了就停下并说清楚缺什么。

    tinygo data --check          # 只检查：每一步要什么、有没有、会产出什么，不跑
    tinygo data sft rl           # 只跑这几步
    tinygo data                  # 全部：download → clean → mix → sft → rl

步骤：
    download  下载全部原始数据                                     dataset/download.py
    clean     按来源清洗预训练数据                                 dataset/pretrain.py
    mix       去重、配比、写出四档预训练文件、检查、生成数据集说明 dataset/mix.py
    sft       SFT 语料（并留出一小份问题给 RL 和评测）             dataset/sft.py
    rl        RL 题目、评测题、MOPD 提示                           dataset/rl.py
"""
import argparse
from pathlib import Path

from dataset.mix import MIX, TIERS, final_path
from dataset.sources import CLEAN, DATA, INPUTS, PRETRAIN, RAW, RL_INPUTS, ROOT, SFT

MIX_SOURCES = [s for _, (_, sources) in MIX.items() for s in sources]


def _have(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(p for p in path.iterdir() if not p.name.startswith(("_", "."))))


def _download(args):
    from dataset import download

    download.main(["pretrain", "sft", "rl", "extra"])


def _clean(args):
    from dataset import pretrain

    for name in PRETRAIN:
        pretrain.clean_source(name, args.workers)


def _mix(args):
    from dataset import mix

    for command in ("index", "plan", "write", "verify", "card"):
        mix.main([command])


def _sft(args):
    from dataset import sft

    sft.main([])


def _rl(args):
    from dataset import rl

    rl.main([])


# 每一步：要什么（必须有的、最好有的）、产出什么、怎么跑
STEPS = {
    "download": dict(needs=[], optional=[], makes=[RAW, INPUTS / "sft", INPUTS / "rl"], run=_download),
    "clean": dict(needs=[RAW / spec.get("raw", name) for name, spec in PRETRAIN.items()] + [RAW / "pg19_heldout"], optional=[],
                  makes=[CLEAN / name for name in PRETRAIN], run=_clean),
    "mix": dict(needs=[CLEAN / s for s in MIX_SOURCES], optional=[], makes=[final_path(t) for t in TIERS] + [ROOT / "out" / "mix" / "README.md"], run=_mix),
    "sft": dict(needs=[INPUTS / "sft" / f"{name}.jsonl" for name in SFT] + [CLEAN / "luxun", CLEAN / "poetry"],
                optional=[INPUTS / "gujilab_translate.jsonl"],
                makes=[DATA / f"sft_{t}.parquet" for t in ("small", "medium", "large")] + [DATA / "sft_heldout.parquet"], run=_sft),
    "rl": dict(needs=[DATA / "sft_large.parquet", DATA / "sft_heldout.parquet", CLEAN / "poetry", CLEAN / "wiki_zh", CLEAN / "luxun"],
               optional=[INPUTS / "rl" / name for name in RL_INPUTS],
               makes=[DATA / f"rl_{t}.parquet" for t in ("small", "medium", "large")] + [DATA / "eval_tasks.parquet", DATA / "mopd_prompts.parquet"],
               run=_rl),
}


def check(step: str, made: set[Path] = frozenset()) -> tuple[list[Path], list[Path]]:
    """这一步缺哪些必须的输入、哪些最好有的输入。made 是前面几步会产出的（这次一起跑时算作有）。"""
    s = STEPS[step]
    covered = lambda p: _have(p) or p in made or any(p.is_relative_to(m) for m in made if m.suffix == "")
    return [p for p in s["needs"] if not covered(p)], [p for p in s["optional"] if not covered(p)]


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main(argv=None):
    parser = argparse.ArgumentParser(description="从头到尾做出全部数据")
    parser.add_argument("steps", nargs="*", choices=list(STEPS), help="只跑这几步（按 STEPS 的顺序）；不写就是全部")
    parser.add_argument("--check", action="store_true", help="只检查输入在不在，不跑")
    parser.add_argument("--workers", type=int, default=8, help="clean 用几个进程")
    args = parser.parse_args(argv)
    steps = [s for s in STEPS if s in (args.steps or STEPS)]
    made, problems = set(), []
    for step in steps:
        missing, weak = check(step, made)
        state = "可以跑" if not missing else f"缺 {len(missing)} 项"
        print(f"[{step}] {state}；产出：{'、'.join(_rel(p) for p in STEPS[step]['makes'])}")
        for p in missing:
            print(f"    缺：{_rel(p)}")
        for p in weak:
            print(f"    没有（可以跑，但少这一部分）：{_rel(p)}")
        if missing:
            problems.append(step)
        made |= set(STEPS[step]["makes"])
    if args.check:
        return problems
    if problems:
        raise SystemExit(f"这几步缺输入，没有开跑：{'、'.join(problems)}（先跑前面的步骤，或者 python -m dataset.download 下载）")
    for step in steps:
        print(f"\n===== {step}", flush=True)
        STEPS[step]["run"](args)
    return []


if __name__ == "__main__":
    main()
