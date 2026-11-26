"""打印一个预设的参数量明细和各部件的占比：tinygo report small"""
import argparse

from lm import PRESETS, LanguageModel, preset


def main(argv=None):
    parser = argparse.ArgumentParser(description="模型参数量明细")
    parser.add_argument("preset", choices=list(PRESETS))
    parser.add_argument("--vocab", type=int, default=32768, help="词表大小（默认是正式分词器的 32768）")
    args = parser.parse_args(argv)
    cfg = preset(args.preset, args.vocab)
    model = LanguageModel(cfg)
    summary, parts = model.param_report(), model.param_breakdown()
    total, trunk = summary["合计"], summary["主干"]
    print(f"{args.preset}：{cfg.dim} 宽 × {cfg.n_layers} 层，词表 {args.vocab}")
    for name, n in summary.items():
        print(f"  {name:<14}{n / 1e6:>10.2f}M{n / total:>8.1%}")
    print("各部件（占合计 / 占主干）：")
    outside = ("词向量", "输出层", "记忆表")
    for name, n in sorted(parts.items(), key=lambda kv: -kv[1]):
        of_trunk = "" if name in outside else f"{n / trunk:>8.1%}"
        print(f"  {name:<14}{n / 1e6:>10.2f}M{n / total:>8.1%}{of_trunk}")


if __name__ == "__main__":
    main()
