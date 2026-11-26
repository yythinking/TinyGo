"""在命令行里和模型对话：

    tinygo chat                 # 不写存档：用 out/small 里走得最远的那个（rl、mopd、sft 依次找）
    tinygo chat --demo          # 先把内置的几个问题挨个问一遍，看看模型现在什么样
    tinygo chat out/small/sft/ckpt.pt --tools

每次回车发一句，模型回答完接着等下一句。输入 /清空 重新开始对话，/退出 结束。
--tools 打开基础工具（计算器、跑 Python），模型写出工具调用时会真的执行，再把结果交给它接着回答。
"""
import argparse
import time

import torch

from training.trainer_utils import find_stage, load_model, pick_device

from .generate import generate
from .template import ChatFormat
from .tokenizer import DEFAULT_MODEL, Tokenizer


# --demo 挨个问的问题：日常、常识、代码、算术、诗词、文言、概念、推荐，每个领域一个
DEMO_QUESTIONS = ["你好，请介绍一下你自己。", "为什么天空是蓝色的？", "用 Python 写一个判断质数的函数。", "小明有 5 个苹果，又买了 3 个，吃掉 2 个，还剩几个？",
                  "写一首关于秋天的五言绝句。", "把「学而时习之，不亦说乎」翻译成白话文。", "用一段话解释什么是机器学习。", "推荐三本适合入门的中国历史书。"]


def reply(model, fmt: ChatFormat, messages: list[dict], max_new: int = 512, temperature: float = 0.7, top_k: int | None = 50,
          env=None, max_turns: int = 4) -> tuple[str, list[dict]]:
    """返回 (最后的回答, 这一问里新增的消息：可能有工具调用和工具结果)。"""
    device = next(model.parameters()).device
    added = []
    for _ in range(max_turns):
        prompt = fmt.prompt(messages + added)
        room = min(max_new, model.cfg.max_seq_len - len(prompt))
        if room <= 0:
            return "（对话太长了，请 /清空 重新开始）", added
        out = generate(model, torch.tensor([prompt], device=device), room, temperature=temperature, top_k=top_k, eos_id=fmt.end)
        text, calls, broken = fmt.parse_turn(out[0, len(prompt):].tolist(),allow_plain_calls=env is not None)
        added.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
        if (not calls and not broken) or env is None:
            return text, added
        results = [env.call(c["name"], c["arguments"]) for c in calls] + [f"工具调用格式错误：{b[:200]}" for b in broken]
        added.append({"role": "tool", "content": "\n".join(results)})
    return added[-1]["content"], added


def main(argv=None, input_fn=input, print_fn=print):
    parser = argparse.ArgumentParser(description="和模型对话")
    parser.add_argument("ckpt", nargs="?", default=None, help="存档；不写就用运行目录里走得最远的那个（rl、mopd、sft 依次找）")
    parser.add_argument("--run", default="out/small", help="运行目录")
    parser.add_argument("--demo", action="store_true", help="把内置的几个问题挨个问一遍，问完就结束")
    parser.add_argument("--system", default="", help="系统提示")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tools", action="store_true", help="打开计算器和 Python 工具")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.ckpt is None:
        args.ckpt = str(find_stage(args.run, ("rl", "mopd", "sft")))
        print_fn(f"用的存档：{args.ckpt}")
    device = pick_device(args.device)
    model = load_model(args.ckpt, device)
    from .artifact import tokenizer_path
    fmt = ChatFormat(Tokenizer(tokenizer_path(args.ckpt, DEFAULT_MODEL)))
    env = None
    if args.tools:
        from training.rewards.tools import ToolEnv

        env = ToolEnv({"tools": ["calculator", "python"]})
    base = [{"role": "system", "content": "\n".join(x for x in (args.system, env.describe() if env else "") if x)}] if (args.system or env) else []
    messages = list(base)
    if args.demo:
        for question in DEMO_QUESTIONS:
            start = time.perf_counter()
            text, _ = reply(model, fmt, base + [{"role": "user", "content": question}], args.max_new, args.temperature, args.top_k, env)
            speed = len(fmt.tok.encode(text)) / max(time.perf_counter() - start, 1e-9)
            print_fn(f"你：{question}")
            print_fn(f"模型：{text}\n（每秒 {speed:.0f} 词）\n")
        return messages
    while True:
        try:
            line = input_fn("你：").strip()
        except EOFError:
            break
        if line in ("/退出", "/exit"):
            break
        if line in ("/清空", "/clear"):
            messages = list(base)
            continue
        if not line:
            continue
        messages.append({"role": "user", "content": line})
        text, added = reply(model, fmt, messages, args.max_new, args.temperature, args.top_k, env)
        messages += added
        print_fn("模型：" + text)
    return messages


if __name__ == "__main__":
    main()
