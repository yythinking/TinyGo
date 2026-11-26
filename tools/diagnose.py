"""诊断：生成的文字打转、不停、和损失对不上时，分清问题出在数据、计算路径还是训练得不够。只读存档，不改任何东西。

    tinygo diagnose                                   # out/small 里的预训练存档，数据取训练时用的那份
    tinygo diagnose out/small/pretrain/ckpt.pt --data data/tokens/pretrain_base
    tinygo diagnose --device cpu                      # 显卡正在训练、不想碰它时（少了 CUDA 上的几项对比）

训练还在跑时也能用：读的是最近一次存下的 ckpt.pt，默认设置下占显存 2 GB 以内。依次检查：

  1. 训练记录    日志里最近的损失、验证集损失、已经训了多少词。
  2. 数据        结束符占多少、一篇平均多长（决定了短续写里该不该出现结束符）；真实文字的重复度，当参照。
  3. 路径一致    同一段词分别走：整段前向（训练用的那条）、逐词缓存、静态缓存（CUDA 上生成用的那条）、CPU 单精度。
                 各条路径给出的损失和下一个词应该一样。不一样，说明训练和生成算的不是同一个函数——内核、精度或缓存有错。
                 还会用训练时验证的那几批片段在这台设备上重算验证集损失，和日志里的数比：在 CPU 上跑诊断时，
                 这一项就是「CPU 单精度」对「CUDA 半精度加各种内核」的直接对比。
                 顺带报告各归一化层输入的大小：nn.RMSNorm 的 ε 默认取输入精度的机器 ε（BF16 是 0.0078，FP32 是 1.2e-7），
                 输入的均方小到和 0.0078 可比时，半精度训练、单精度评测会得到不同的结果。
  4. 因果性      改掉后半段的词，前半段的输出不能变。变了就是训练时偷看了后面的词：损失很低，生成一塌糊涂。
  5. 用上下文    损失按片段里的位置分段。前面的词越多损失应该越低；一路持平说明注意力没起作用，模型只靠最近两三个词。
                 同时比较「模型放在前 50 名以外的概率」和「真实答案落在前 50 名以外的比例」，看概率有没有撒到不相干的词上。
  6. 结束符      真正的文档结尾处，模型给结束符的概率，对比别处。
  8. 尾部分语言  在纯中文的上文里，模型给拉丁字母片段、假名这些词多少概率，对比它们实际出现的比例。总量对得上（第 5 项）
                 不代表成分对：模型可能知道「这里有一成机会出一个生僻词」，却不知道该是哪一类。差得多时，不截断的采样会抽出外文片段。
  7. 采样        同一个开头（前面放文档结束符，和 tinygo generate 一样），贪心、默认采样、温度 1 各续写一次，报告 4 词片段的重复比例。
"""
import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import torch

from inference.fast import StaticState, forward_static
from inference.generate import DEMO_PROMPTS, generate, prompt_ids
from inference.tokenizer import DEFAULT_MODEL, Tokenizer
from lm.layers.norm import RMSNorm
from lm.loss import token_log_probs
from training.dataset import MixedData, PackedData, open_mixed
from training.trainer_utils import autocast, count, find_stage, load_model, pick_device, setup_device

LOSS_TOLERANCE = 0.05   # 两条路径的损失差超过它就算不一致（半精度和单精度之间通常差 0.01 以内）
AGREE_TOLERANCE = 0.95  # 两条路径选出的下一个词，一致的比例低于它就算不一致
LEAK_TOLERANCE = 0.05   # 改了后面的词，前面的输出变化超过它就算偷看


def repeat_share(ids: list[int], n: int = 4) -> float:
    """n 个词的片段里，有多大比例是前面出现过的。真实文字一般在 0.05 以下，打转的文字接近 1。"""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / max(1, len(grams))


KINDS = ("汉字", "拉丁字母", "假名、韩文、西里尔字母", "数字、标点等")
DEEP = 1e-4  # 概率低于它的词算「深尾」：温度 1.0 不截断时，怪词主要从这里抽出来


def token_kinds(tok: Tokenizer) -> torch.Tensor:
    """词表里每个词属于哪一类文字（KINDS 的下标）。"""
    def kind(piece: str) -> int:
        if re.search(r"[\u4e00-\u9fff]", piece):
            return 0
        if re.search(r"[\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]", piece):
            return 2
        return 1 if re.search(r"[A-Za-z]", piece) else 3
    return torch.tensor([kind(tok.sp.id_to_piece(i)) for i in range(tok.vocab_size)])


def chinese_context(kinds_of_input: torch.Tensor, span: int = 16) -> torch.Tensor:
    """哪些位置的前 span 个词（含自己）全是汉字或标点、且至少一半是汉字。形状同输入（一维）。"""
    out = torch.zeros_like(kinds_of_input, dtype=torch.bool)
    if kinds_of_input.numel() >= span:
        allowed = ((kinds_of_input == 0) | (kinds_of_input == 3)).float().unfold(0, span, 1).mean(-1) == 1
        mostly = (kinds_of_input == 0).float().unfold(0, span, 1).mean(-1) >= 0.5
        out[span - 1:] = allowed & mostly
    return out


def shards_of(data) -> list:
    part = data.parts[0] if isinstance(data, MixedData) else data
    if isinstance(part, PackedData):
        raise ValueError("诊断只支持预训练的词数据（一长串词），不支持打包好的对话数据")
    return part.shards


def doc_end_windows(shards: list, eos: int, length: int, rows: int, scan: int = 20_000_000) -> torch.Tensor:
    """取 rows 个长 length+1 的片段，每个都恰好结束在一个文档结束符上。形状 (rows, length+1)。"""
    out = []
    for shard in shards:
        head = np.asarray(shard[:scan])
        ends = np.flatnonzero(head == eos)
        ends = ends[ends >= length]
        for e in ends[np.linspace(0, len(ends) - 1, min(rows, len(ends))).astype(int)] if len(ends) else []:
            out.append(head[e - length:e + 1].astype(np.int64))
        if len(out) >= rows:
            break
    if not out:
        raise ValueError(f"数据的前 {count(scan)} 词里没有找到前面够 {length} 个词的文档结束符（编号 {eos}）")
    return torch.from_numpy(np.stack(out[:rows]))


@torch.no_grad()
def full_forward(model, x: torch.Tensor, half: bool) -> torch.Tensor:
    """整段前向，返回 (B, T, 词表) 的单精度输出（放在 CPU 上）。half：CUDA 上用 BF16（训练就是这样算的）。"""
    device = x.device
    with autocast(device) if half else torch.autocast(device.type, enabled=False):
        return model(x, return_logits=True).logits.float().cpu()


@torch.no_grad()
def stepwise_forward(model, x: torch.Tensor) -> torch.Tensor:
    """逐词缓存：inference.generate 的普通路径（不开半精度）。"""
    state, out = model.new_state(), []
    for t in range(x.size(1)):
        out.append(model(x[:, t:t + 1], state=state).logits[:, -1].float().cpu())
    return torch.stack(out, dim=1)


@torch.no_grad()
def static_forward(model, x: torch.Tensor) -> torch.Tensor:
    """静态缓存：inference.fast 的路径（CUDA 上 generate 默认走它，开 BF16）。"""
    device = x.device
    half = device.type == "cuda"
    state, out = StaticState(model, x.size(0), torch.bfloat16 if half else torch.float32, device), []
    with autocast(device):
        for t in range(x.size(1)):
            out.append(forward_static(model, x[:, t:t + 1], torch.tensor([t], device=device), state).float().cpu())
    return torch.stack(out, dim=1)


@torch.no_grad()
def validation_loss(model, data, batch_size: int, seq_len: int, batches: int, device: torch.device, rows_at_once: int = 4) -> float:
    """和 train_pretrain.evaluate 取完全相同的片段（同样的种子），在这台设备上重算验证集损失。一次只算几段，CPU 上也放得下。"""
    total, n = 0.0, 0
    for i in range(batches):
        x, y = data.batch(batch_size, seq_len, seed=1_000_003 * (i + 1), device=device)
        for s in range(0, batch_size, rows_at_once):
            with autocast(device):
                out = model(x[s:s + rows_at_once], y[s:s + rows_at_once], return_logits=False)
            rows = x[s:s + rows_at_once].size(0)
            total += float(out.loss - out.balance_loss) * rows
            n += rows
        print(f"\r  重算验证集损失：{i + 1}/{batches} 批，目前 {total / n:.4f}", end="", flush=True)
    print()
    return total / n


def norm_inputs(model, x: torch.Tensor) -> list[tuple[str, float]]:
    """每个归一化层输入的均方（取 1% 分位：最小的那批位置），从小到大。"""
    seen, hooks = {}, []
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            def hook(_module, inputs, _output, name=name):
                ms = inputs[0].detach().float().pow(2).mean(-1).flatten()
                seen[name] = float(ms.kthvalue(max(1, ms.numel() // 100)).values)
            hooks.append(module.register_forward_hook(hook))
    full_forward(model, x, half=True)
    for h in hooks:
        h.remove()
    return sorted(seen.items(), key=lambda kv: kv[1])


def diagnose(args) -> dict:
    device = pick_device(args.device)
    setup_device(device)
    ckpt = Path(args.ckpt or find_stage(args.run, ("pretrain", "anneal")))
    config = json.loads((ckpt.parent / "config.json").read_text(encoding="utf-8")) if (ckpt.parent / "config.json").exists() else {}
    train_args = config.get("train", {})
    report, problems = {}, []

    try:
        import liger_kernel
        liger = getattr(liger_kernel, "__version__", "已装")
    except ImportError:
        liger = "没装"
    print(f"存档 {ckpt}；设备 {device}；PyTorch {torch.__version__}；liger-kernel {liger}")

    # 1. 训练记录
    print("\n[1] 训练记录")
    log = ckpt.parent / "log.jsonl"
    if log.exists():
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        steps = config.get("steps")
        last = next((r for r in reversed(records) if "smooth_loss" in r), {})
        val = [r for r in records if "val_loss" in r]
        report["val_log"] = val[-1] if val else None
        print(f"  最近一条：第 {last.get('step')} 步" + (f" / 共 {steps} 步（{last.get('step', 0) / steps:.1%}）" if steps else "") +
              f"，已训 {count(last.get('tokens', 0))} 词，平滑损失 {last.get('smooth_loss')}，学习率×{last.get('lr_scale')}")
        if val:
            print("  验证集损失（最近几次）：" + "  ".join(f"第{r['step']}步 {r['val_loss']}" for r in val[-6:]))
            if "saturated" in val[-1]:
                print(f"  softmax 塌缩的词占比：{val[-1]['saturated']:.2%}")
        report["log"] = last
    else:
        print(f"  {log} 不存在")
    if train_args:
        keys = ("preset", "seq_len", "batch_size", "grad_accum", "lr_mult", "matrix_lr", "adam_eps", "cuda_graph", "no_compile")
        print("  训练参数：" + "，".join(f"{k}={train_args[k]}" for k in keys if k in train_args))

    model = load_model(ckpt, device).eval()
    cfg = model.cfg
    data_spec = args.data or train_args.get("data")
    if not data_spec:
        raise ValueError("存档旁边的 config.json 里没有数据路径，用 --data 指定切好的词数据目录")
    data = open_mixed(data_spec, "val") or open_mixed(data_spec, "train")
    eos = data.meta["eos_id"]
    shards = shards_of(data)
    length = min(args.seq_len or train_args.get("seq_len", 2048), cfg.max_seq_len)
    tok = Tokenizer(DEFAULT_MODEL)

    # 2. 数据
    print("\n[2] 数据")
    head = np.asarray(shards_of(open_mixed(data_spec, "train"))[0][:20_000_000])
    eos_share = float((head == eos).mean())
    print(f"  训练数据前 {count(head.size)} 词：结束符（编号 {eos}）占 {eos_share:.4%}，一篇平均 {1 / max(eos_share, 1e-9):,.0f} 词；"
          f"也就是说，续写 100 个词，就算模型学得完美，出现结束符的机会也只有 {1 - (1 - eos_share) ** 100:.1%}")
    real = [repeat_share(head[s:s + 120].tolist()) for s in range(0, min(head.size - 120, 1_200_000), 12_000)]
    print(f"  真实文字（120 词一段）的 4 词重复比例：平均 {np.mean(real):.3f}，最高 {np.max(real):.3f}")
    print("  随便看一段训练数据：" + repr(tok.decode(head[5000:5120].tolist()))[:260])
    report["eos_share"] = eos_share

    # 3. 路径一致
    rows = doc_end_windows(shards, eos, length, args.rows).to(device)
    x, y = rows[:1, :-1], rows[:1, 1:].cpu()
    print(f"\n[3] 路径一致（同一段 {length} 个词，各条路径的输出和第一条比）")
    paths = {("整段前向 BF16（训练用的路径）" if device.type == "cuda" else "整段前向 FP32（训练用的路径）"): full_forward(model, x, half=True)}
    if device.type == "cuda":
        paths["整段前向 FP32"] = full_forward(model, x, half=False)
    paths["逐词缓存 FP32（generate 的普通路径）"] = stepwise_forward(model, x)
    paths["静态缓存" + (" BF16（CUDA 上 generate 默认走它）" if device.type == "cuda" else " FP32")] = static_forward(model, x)
    if device.type != "cpu" and not args.no_cpu:
        cpu_model = copy.deepcopy(model).cpu()
        paths["CPU 整段前向 FP32（不经过任何 CUDA 内核）"] = full_forward(cpu_model, x.cpu(), half=False)
        del cpu_model
    base = next(iter(paths.values()))
    base_loss = float(-token_log_probs(base, y).mean())
    report["paths"] = {}
    for name, logits in paths.items():
        loss = float(-token_log_probs(logits, y).mean())
        agree = float((logits.argmax(-1) == base.argmax(-1)).float().mean())
        gap = float((logits - base).abs().max())
        bad = abs(loss - base_loss) > LOSS_TOLERANCE or agree < AGREE_TOLERANCE
        print(f"  {'✗' if bad else '✓'} {name:<34} 损失 {loss:.4f}  下一个词和第一条一致 {agree:.2%}  输出最大差 {gap:.3f}")
        report["paths"][name] = dict(loss=loss, agree=agree, max_gap=gap)
        if bad:
            problems.append(f"「{name}」和训练用的路径算出来的不一样（损失 {loss:.3f} 对 {base_loss:.3f}，下一个词一致 {agree:.1%}）")
    val_batches = train_args.get("eval_batches", 20) if args.val_batches is None else args.val_batches
    logged = report.get("val_log")
    if val_batches and logged and open_mixed(data_spec, "val") and "batch_size" in train_args:
        print(f"  用训练时验证的那几批片段（{val_batches} 批 × {train_args['batch_size']} 段 × {train_args['seq_len']} 词，种子相同）在这台设备上重算：")
        here = validation_loss(model, open_mixed(data_spec, "val"), train_args["batch_size"], train_args["seq_len"], val_batches, device)
        gap = here - logged["val_loss"]
        bad = abs(gap) > LOSS_TOLERANCE
        print(f"  {'✗' if bad else '✓'} 这台设备 {here:.4f}；训练日志里第 {logged['step']} 步是 {logged['val_loss']}（训练的设备和精度算的）；差 {gap:+.4f}"
              + ("" if val_batches == train_args.get("eval_batches", 20) else "（批数和训练时不同，只能粗比）"))
        report["val_here"] = here
        if bad:
            problems.append(f"同样的验证片段，这台设备算出来的损失是 {here:.3f}，训练时记的是 {logged['val_loss']}：两边算的不是同一个函数")
    smallest = norm_inputs(model, x)[:3]
    half_eps = torch.finfo(torch.bfloat16).eps
    print("  归一化层输入的均方（最小的 1% 位置），最小的三层：" + "；".join(f"{n} {v:.3f}" for n, v in smallest))
    print(f"  按最小的那层算，BF16（ε={half_eps:g}）和 FP32（ε≈1.2e-7）归一化出来的大小差 {(1 + half_eps / max(smallest[0][1], 1e-12)) ** 0.5 - 1:.2%}"
          "（1% 以内可以不管）")
    report["norm_inputs"] = smallest

    # 4. 因果性
    print("\n[4] 因果性（改掉后半段的词，前半段的输出不该变）")
    cut = length // 2
    changed = x.clone()
    changed[:, cut:] = torch.randint(4, cfg.vocab_size, changed[:, cut:].shape, device=device)
    leak = float((full_forward(model, changed, half=True)[:, :cut] - base[:, :cut]).abs().max())
    print(f"  {'✗' if leak > LEAK_TOLERANCE else '✓'} 前半段输出的最大变化：{leak:.4f}")
    report["leak"] = leak
    if leak > LEAK_TOLERANCE:
        problems.append(f"训练用的路径会偷看后面的词（前半段的输出变了 {leak:.3f}）：损失是假的")

    # 5. 用上下文 + 6. 结束符
    token_loss, eos_at_end, eos_elsewhere, eos_rank = [], [], [], []
    tail_mass, tail_hits = 0.0, 0
    kinds = token_kinds(tok)
    zeros = lambda: torch.zeros(len(KINDS), dtype=torch.float64)  # noqa: E731
    kind_mass, kind_hits, deep_mass, deep_hits, chinese_n = zeros(), zeros(), zeros(), zeros(), 0
    for r in range(rows.size(0)):
        logits = base if r == 0 else full_forward(model, rows[r:r + 1, :-1], half=True)
        target = rows[r:r + 1, 1:].cpu()
        token_loss.append(-token_log_probs(logits, target)[0])
        probs = torch.softmax(logits[0], dim=-1)
        p_eos = probs[:, eos]
        top = probs.topk(50, dim=-1)
        tail_mass += float((1 - top.values.sum(-1)).sum())
        tail_hits += int((~(top.indices == target[0][:, None]).any(-1)).sum())
        pure = chinese_context(kinds[rows[r, :-1].cpu()])
        pp, tt = probs[pure], target[0][pure]
        deep = pp * (pp < DEEP)
        true_is_deep = pp.gather(1, tt[:, None])[:, 0] < DEEP
        for k in range(len(KINDS)):
            kind_mass[k] += float(pp[:, kinds == k].sum())
            deep_mass[k] += float(deep[:, kinds == k].sum())
        kind_hits += torch.bincount(kinds[tt], minlength=len(KINDS))
        deep_hits += torch.bincount(kinds[tt[true_is_deep]], minlength=len(KINDS))
        chinese_n += int(pure.sum())
        is_end = target[0] == eos
        eos_at_end += p_eos[is_end].tolist()
        eos_elsewhere += p_eos[~is_end].tolist()
        eos_rank += (logits[0][is_end] > logits[0][is_end][:, eos:eos + 1]).sum(-1).tolist()
    token_loss = torch.stack(token_loss)
    print(f"\n[5] 用上下文（{rows.size(0)} 段，按在片段里的位置分段的平均损失）")
    edges = [e for e in (0, 4, 32, 128, 512, 1024) if e < length] + [length]
    buckets = [float(token_loss[:, a:b].mean()) for a, b in zip(edges, edges[1:])]
    print("  " + "   ".join(f"{a}–{b}: {v:.3f}" for a, b, v in zip(edges, edges[1:], buckets)))
    report["loss_by_position"] = dict(zip([f"{a}-{b}" for a, b in zip(edges, edges[1:])], buckets))
    if length > 128:  # 只有最近两三个词可用的模型，第 4 个词以后损失就不再降了
        near, far = float(token_loss[:, 4:32].mean()), float(token_loss[:, 128:].mean())
        if far > near - 0.1:
            problems.append(f"前面的词再多，损失也不降（第 4–32 个词 {near:.2f}，第 128 个词以后 {far:.2f}）：模型没在用远处的上下文")

    tail_model, tail_real = tail_mass / token_loss.numel(), tail_hits / token_loss.numel()
    print(f"  前 50 名以外：真实的下一个词落在这里的占 {tail_real:.1%}，模型放在这里的概率是 {tail_model:.1%}。"
          "两个数接近，说明模型没有把概率乱撒；这一块就是温度 1.0 不截断时怪词的来源，也是只取前 50 时被丢掉的部分")
    report["tail"] = dict(model=tail_model, real=tail_real)
    if tail_model > 1.5 * tail_real + 0.02:
        problems.append(f"模型放在前 50 名以外的概率（{tail_model:.1%}）比真实答案落在那里的比例（{tail_real:.1%}）高得多：概率撒到了不相干的词上")

    print(f"\n[6] 结束符（{len(eos_at_end)} 个真正的文档结尾）")
    print(f"  结尾处给结束符的概率：中位 {np.median(eos_at_end):.3%}，平均 {np.mean(eos_at_end):.3%}；别处平均 {np.mean(eos_elsewhere):.4%}"
          f"（数据里的底数是 {eos_share:.4%}）")
    print(f"  结尾处结束符排第几：中位第 {int(np.median(eos_rank)) + 1} 名，排进前 5 的占 {np.mean(np.array(eos_rank) < 5):.0%}")
    report["eos"] = dict(at_end=float(np.mean(eos_at_end)), elsewhere=float(np.mean(eos_elsewhere)))

    # 7. 采样
    print(f"\n[8] 尾部分不分语言（{chinese_n} 个纯中文上文的位置：前 16 个词全是汉字或标点）")
    if chinese_n:
        print("  下一个词的类别        真实占比   模型给的概率   倍数")
        for k, name in enumerate(KINDS):
            ratio = f"{kind_mass[k] / kind_hits[k]:8.2f}" if kind_hits[k] >= 20 else "  （太少）"
            print(f"  {name:<12} {kind_hits[k] / chinese_n:9.3%} {kind_mass[k] / chinese_n:11.3%} {ratio}")
        foreign_model = float(kind_mass[1] + kind_mass[2]) / chinese_n
        foreign_real = float(kind_hits[1] + kind_hits[2]) / chinese_n
        deep_model, deep_real = float(deep_mass.sum()), max(float(deep_hits.sum()), 1.0)
        print(f"  深尾（概率低于 {DEEP:g} 的词）：模型放了 {deep_model / chinese_n:.2%}，真实答案落在这里 {float(deep_hits.sum()) / chinese_n:.2%}；"
              f"其中不是中文的：模型 {float(deep_mass[1] + deep_mass[2]) / max(deep_model, 1e-12):.1%}，真实 {float(deep_hits[1] + deep_hits[2]) / deep_real:.1%}")
        report["script"] = dict(foreign_model=foreign_model, foreign_real=foreign_real, positions=chinese_n,
                                deep_foreign_model=float(deep_mass[1] + deep_mass[2]) / max(deep_model, 1e-12),
                                deep_foreign_real=float(deep_hits[1] + deep_hits[2]) / deep_real)
        deep_foreign = report["script"]["deep_foreign_model"]
        if deep_foreign > 3 * report["script"]["deep_foreign_real"] + 0.05:
            problems.append(f"纯中文的上文里，模型深尾的概率有 {deep_foreign:.0%} 放在非中文的词上，真实答案落在深尾时只有 "
                            f"{report['script']['deep_foreign_real']:.0%} 不是中文：尾部的总量对，成分不对。"
                            "不截断的采样会抽出外文片段；用 top_p 或 top_k 截掉尾部可以避开")

    print(f"\n[7] 采样（每个开头续写 {args.max_new} 个词；后面的数是 4 词重复比例）")
    settings = [("贪心", dict(temperature=0.0)), ("温度0.8 前50", dict(temperature=0.8, top_k=50)), ("温度1.0", dict(temperature=1.0))]
    report["samples"] = {}
    for prompt in DEMO_PROMPTS[:3]:
        ids = torch.tensor([prompt_ids(tok, prompt)], device=device)
        for name, kw in settings:
            torch.manual_seed(args.seed)
            out = generate(model, ids, args.max_new, eos_id=None, **kw)[0, ids.size(1):].tolist()
            share = repeat_share(out)
            report["samples"][f"{prompt}|{name}"] = share
            print(f"  [{name:<9} {share:.2f}] {prompt}{tok.decode(out)!r}"[:args.width])

    print("\n[结论]")
    if problems:
        for p in problems:
            print("  ✗ " + p)
    else:
        print("  ✓ 各条计算路径一致、没有偷看、上下文有用、尾部成分正常：生成和训练算的是同一个函数。")
        print("    这只说明代码没有算错，不说明续写质量够好：贪心打转、接词不合事实要另外看 [7] 的文字。")
    report["problems"] = problems
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="诊断：生成重复、不停时，分清是数据、计算路径还是训练不够")
    p.add_argument("ckpt", nargs="?", default=None, help="存档；不写就用运行目录里的预训练存档")
    p.add_argument("--run", default="out/small", help="运行目录")
    p.add_argument("--data", default=None, help="切好的词数据目录；不写就用存档旁边 config.json 里记的那份")
    p.add_argument("--seq-len", type=int, default=0, help="检查用多长的片段；不写就用训练时的长度")
    p.add_argument("--val-batches", type=int, default=None, help="重算验证集损失用几批；不写就和训练时一样（这样才能和日志里的数直接比），0 是跳过")
    p.add_argument("--rows", type=int, default=16, help="第 5、6 项用几段")
    p.add_argument("--max-new", type=int, default=100, help="第 7 项每个开头续写多少个词")
    p.add_argument("--no-cpu", action="store_true", help="跳过 CPU 上的那次对比（大模型在 CPU 上很慢）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=230, help="第 7 项每行最多打印多少个字符")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None):
    return diagnose(parse_args(argv))


if __name__ == "__main__":
    main()
