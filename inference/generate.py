"""逐词生成。第一步把整段提示送进去并建好缓存，之后每步只算新来的一个词。

    tinygo generate                                  # 不写存档：用 out/small 里的退火（没有就预训练）存档，续写几个内置的开头
    tinygo generate out/small/pretrain/ckpt.pt --prompt "我家门前有两株树，" --max-new 100

命令行默认采样（温度 0.8、只在概率最高的 50 个词里挑）。--temperature 0 是每步取概率最大的词：小模型这样很容易陷进重复。
"""
import argparse
import weakref
from collections.abc import Iterable

import torch
import torch.nn.functional as F

from lm import LanguageModel
from training.trainer_utils import find_stage, load_model, pick_device

from .tokenizer import DEFAULT_MODEL, Tokenizer


DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50


def sampling_generator(device: torch.device, seed: int) -> torch.Generator | None:
    """CPU、CUDA 都使用对应设备的随机数生成器，让固定种子的评测可复现。"""
    return torch.Generator(device=device).manual_seed(seed) if device.type in ("cpu", "cuda") else None


def prompt_ids(tok: Tokenizer, prompt: str) -> list[int]:
    """续写用的提示：开头放一个文档结束符，意思是「上一篇完了，这里是新一篇的开头」。
    训练数据里每篇的开头前面都是它；不放的话，模型会把提示当成从某篇文章中间截出来的一段，短提示尤其容易接歪。"""
    return [tok.eos_id] + tok.encode(prompt)


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits: (B, 词表) → (B, 1) 选中的词。temperature 为 0 时直接取最大的。"""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        values, indices = logits.topk(min(top_k, logits.size(-1)), dim=-1)
        logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, values)
    if top_p is not None:
        ordered, order = logits.sort(dim=-1, descending=True)
        probs = F.softmax(ordered, dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p  # 排在前面的累计概率已经超过 top_p 的都不要
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)


_FAST = weakref.WeakKeyDictionary()  # 每个模型一个 FastGenerator，编译过的那一步只编一次


@torch.no_grad()
def generate(model: LanguageModel, idx: torch.Tensor, max_new_tokens: int, temperature: float = DEFAULT_TEMPERATURE, top_k: int | None = DEFAULT_TOP_K,
             top_p: float | None = None, eos_id: int | Iterable[int] | None = None, generator: torch.Generator | None = None,
             fast: bool | None = None) -> torch.Tensor:
    """idx: (B, T) 的提示。返回提示加上新生成的词。到了长度上限，或者每一行都出过停止符（eos_id，可以给一个或几个），就停。
    已经停下的行后面还会接着采样（为了整批一起算），用的时候按第一个停止符截断。
    fast：用预分配缓存 + 编译的快速路径（inference.fast）；不指定时 CUDA 上用，别的设备不用。两条路径结果一样。"""
    if fast if fast is not None else idx.device.type == "cuda":
        from .fast import FastGenerator

        gen = _FAST.get(model)
        if gen is None:
            gen = _FAST[model] = FastGenerator(model)
        return gen.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=eos_id, generator=generator)
    was_training = model.training
    model.eval()
    state = model.new_state()
    done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    stops = None if eos_id is None else torch.tensor([eos_id] if isinstance(eos_id, int) else list(eos_id), device=idx.device)
    step = idx
    for _ in range(min(max_new_tokens, model.cfg.max_seq_len - idx.size(1))):
        logits = model(step, state=state).logits[:, -1]
        step = sample_next(logits, temperature, top_k, top_p, generator)
        idx = torch.cat([idx, step], dim=1)
        if stops is not None:
            done |= torch.isin(step.squeeze(1), stops)
            if done.all():
                break
    model.train(was_training)
    return idx


@torch.no_grad()
def stream(model: LanguageModel, idx: torch.Tensor, max_new_tokens: int, temperature: float = DEFAULT_TEMPERATURE, top_k: int | None = DEFAULT_TOP_K,
           top_p: float | None = None, eos_id: int | Iterable[int] | None = None, generator: torch.Generator | None = None):
    """和 generate 的普通路径一样逐词生成（只收一行提示），但每出一个词就交出来，网页上边写边显示。
    写出停止符（也会交出来）或到了长度上限就停；用的一方不再要了（关掉这个生成器），就不再往下算。"""
    if idx.size(0) != 1:
        raise ValueError("stream 一次只生成一行")
    was_training = model.training
    model.eval()
    try:
        state = model.new_state()
        stops = set() if eos_id is None else ({eos_id} if isinstance(eos_id, int) else set(eos_id))
        step = idx
        for _ in range(min(max_new_tokens, model.cfg.max_seq_len - idx.size(1))):
            logits = model(step, state=state).logits[:, -1]
            step = sample_next(logits, temperature, top_k, top_p, generator)
            token = int(step)
            yield token
            if token in stops:
                return
    finally:
        model.train(was_training)


# 不给 --prompt 时续写的开头：诗词、常识、代码、英文
DEMO_PROMPTS = ["床前明月光，", "北京是中国的", "人工智能是", "def fibonacci(n):", "The capital of France is"]


def _continue(model, tok: Tokenizer, prompt: str, max_new: int, temperature: float, top_k: int | None, seed: int) -> str:
    dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids(tok, prompt)], device=dev)
    gen = sampling_generator(dev, seed) if temperature else None
    out = generate(model, ids, max_new, temperature=temperature, top_k=top_k, eos_id=tok.eos_id, generator=gen)
    return tok.decode(out[0, ids.size(1):].tolist())


def continue_text(ckpt: str, prompt: str, max_new: int = 100, temperature: float = DEFAULT_TEMPERATURE, top_k: int | None = DEFAULT_TOP_K,
                  device: str = "auto", tokenizer: str | None = None, seed: int = 0) -> str:
    """返回模型接在 prompt 后面写出的文字（不含 prompt 本身）。默认在前 50 名内按温度 0.8 采样；temperature=0 才每步取概率最大的词。"""
    model = load_model(ckpt, pick_device(device))
    from .artifact import tokenizer_path
    return _continue(model, Tokenizer(tokenizer or tokenizer_path(ckpt, DEFAULT_MODEL)), prompt, max_new, temperature, top_k, seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="从存档续写文字")
    parser.add_argument("ckpt", nargs="?", default=None, help="存档；不写就用运行目录里的退火存档，没有就用预训练的")
    parser.add_argument("--run", default="out/small", help="运行目录")
    parser.add_argument("--prompt", default=None, help="开头；不写就续写几个内置的开头")
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="默认 0.8；0 是每步取概率最大的词（小模型容易陷进重复）")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="只在概率最高的 k 个词中采样，默认 50")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ckpt = args.ckpt or str(find_stage(args.run, ("anneal", "pretrain")))
    if args.ckpt is None:
        print(f"用的存档：{ckpt}")
    from .artifact import tokenizer_path
    model, tok = load_model(ckpt, pick_device(args.device)), Tokenizer(tokenizer_path(ckpt, DEFAULT_MODEL))
    texts = []
    for prompt in [args.prompt] if args.prompt else DEMO_PROMPTS:
        texts.append(_continue(model, tok, prompt, args.max_new, args.temperature, args.top_k, args.seed))
        print(prompt + texts[-1] + ("\n" if not args.prompt else ""))
    return texts[0] if args.prompt else texts


if __name__ == "__main__":
    main()
