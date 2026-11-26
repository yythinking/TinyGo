"""tinygo 命令：pip install -e . 之后，tinygo <命令> 就等于 python -m <模块>，参数原样交过去。

    tinygo                    # 列出全部命令
    tinygo pretrain --help    # 某个命令的全部参数
"""
import importlib
import sys

# 命令 → (模块, 一句话说明)。顺序就是 tinygo 列出来的顺序：大致是从做数据到用模型的先后
COMMANDS = {
    "data": ("dataset.build", "做数据：下载、清洗、去重分档、SFT 语料、RL 题目（--check 先看缺什么）"),
    "prepare": ("tools.prepare", "切词；训练时给 parquet 会自动切，一般不用单独跑"),
    "tokenizer": ("tools.tokenizer", "抽样本、训练、评测分词器，导出 HF 格式"),
    "pretrain": ("training.train_pretrain", "预训练，也用来退火"),
    "sft": ("training.train_sft", "SFT：通用和领域专家"),
    "distill": ("training.train_distill", "MOPD：多老师在线蒸馏"),
    "rl": ("training.train_rl", "强化学习（GRPO）：RLVR、agent RL、RLAIF"),
    "generate": ("inference.generate", "续写：看预训练模型学到了什么"),
    "chat": ("inference.chat", "命令行对话（--demo 把内置的问题问一遍）"),
    "serve": ("inference.serve", "聊天网页和 OpenAI 兼容的服务"),
    "eval": ("eval.suite", "评测：留出集损失、各题型正确率、评审打分"),
    "benchmarks": ("eval.benchmarks", "C-Eval、CMMLU、GSM8K"),
    "bench": ("tools.bench", "测这张卡的训练速度和显存"),
    "report": ("tools.report", "参数量报告"),
    "overfit": ("tools.overfit", "过拟合检查：在一小批数据上把损失训到接近 0"),
    "diagnose": ("tools.diagnose", "诊断：生成重复、和损失对不上时，分清是数据、计算路径还是训练不够"),
    "demo": ("tools.demo", "在一个 2300 万参数的小模型上把整条链跑一遍"),
    "tensorboard": ("tools.tensorboard", "把训练日志同步成 TensorBoard"),
    "export": ("tools.export", "导出包含配置和分词器的 safetensors 模型"),
    "judge-check": ("tools.judge_check", "检查评审模型打分靠不靠谱"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["用法：tinygo <命令> [参数]，每个命令的全部参数见 tinygo <命令> --help", ""]
    lines += [f"  tinygo {name:<{width}}  {what}" for name, (_, what) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"没有 {name} 这个命令。\n\n{usage()}", file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(COMMANDS[name][0])
    program = sys.argv[0]
    sys.argv[0] = f"tinygo {name}"  # argparse 拿它当用法里的程序名
    try:
        module.main(rest)
    finally:
        sys.argv[0] = program
    return 0  # 命令行入口会把返回值当退出码，各个 main 返回的统计不往外传


if __name__ == "__main__":
    raise SystemExit(main())
