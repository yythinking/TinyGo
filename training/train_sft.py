"""SFT：从一个存档出发，在打包好的对话数据上接着训，只在回答上算损失。三档的通用 SFT 和三个领域专家都用它。

    tinygo sft                                        # 默认：data/sft_small.parquet，从 out/small 里的退火（没有就预训练）出发
    tinygo sft --data data/sft_expert_math.parquet    # 领域专家：从同一个起点出发，写到 out/small/expert_math
    tinygo sft --run out/medium --init out/medium/sft/ckpt.pt --data data/sft_expert_code.parquet

--data 可以直接给 parquet，第一次用时按 --seq-len 打包，放在它旁边的 tokens/ 目录里；也可以给 tools.prepare --chat 打包好的目录。
不写 --out 时：数据叫 sft_expert_<领域> 就写到 <运行目录>/expert_<领域>，否则写到 <运行目录>/sft。

和预训练不同的地方：
  - 模型设置取自出发的存档（--init，不写就在运行目录里找退火或预训练的存档）；
  - 数据是 tools.prepare --chat 打包的对话，一行一个片段，片段长度就是打包时的 --seq-len；遮罩为 0 的位置（提示、填充）不算损失；
  - 按轮数（--epochs）或步数（--steps）训；
  - 学习率默认是预训练的 0.1 倍、预热短、最后降到 0（还没在我们的模型上调过）。
全部参数见文件末尾的 parse_args。
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from .dataset import PackedData, open_data
from .trainer_utils import (add_optimizer_args, autocast, clip_gradients, compile_model, count, find_stage, format_eta, load_checkpoint,
                            load_model, lr_scale, optimizers_from_args, pick_device, resume, restore_rng, rng_state,
                            saturated_share, save_checkpoint, set_lr_scale, setup_device, step_optimizers)


def open_chat_data(directory: str) -> tuple[PackedData, PackedData]:
    data = open_data(directory, "train")
    if not isinstance(data, PackedData):
        raise ValueError(f"{directory} 不是对话数据；先用 python -m tools.prepare --chat 准备")
    if not data:
        raise ValueError(f"{directory} 里没有训练数据")
    return data, PackedData(directory, "val")


def total_steps(args, data: PackedData) -> int:
    """给了 --steps 就用它；否则按 --epochs 算：每轮把全部行打乱后过一遍。"""
    if args.steps:
        return args.steps
    return max(1, math.ceil(args.epochs * data.rows / (args.batch_size * args.grad_accum)))


@torch.no_grad()
def evaluate(model, data: PackedData, args, device: torch.device) -> tuple[float, float]:
    """验证集上回答部分的平均损失。每次用同一批行，前后才可比。
    第二个返回值是第一行里 softmax 塌缩的词占比（见 trainer_utils.saturated_share）。"""
    was_training = model.training
    model.eval()
    total, targets, saturated = 0.0, 0, 0.0
    for i in range(args.eval_batches):
        x, y = data.batch(args.eval_batch_size or args.batch_size, data.row_len - 1,
                          seed=1_000_003 * (i + 1), device=device)
        micro = args.eval_micro_batch_size or args.batch_size
        for start in range(0, len(x), micro):
            xm, ym = x[start:start + micro], y[start:start + micro]
            n = int((ym != -1).sum())
            if not n:
                continue
            with autocast(device):
                result = model(xm, ym, return_logits=False)
            total += (result.loss - result.balance_loss).item() * n
            targets += n
            if i == 0 and start == 0:
                saturated = saturated_share(model, xm, ym, device)
    model.train(was_training)
    if not targets or not math.isfinite(total):
        raise RuntimeError('SFT validation has no finite supervised targets')
    return total / targets, saturated


def train(args) -> dict:
    """跑完 SFT，返回 {"step", "loss", "val_loss", "tokens"}。tokens 只算回答部分的词。"""
    resolve_init(args)
    device = pick_device(args.device)
    setup_device(device)
    if args.gpu_memory_fraction:
        if device.type != 'cuda' or not 0 < args.gpu_memory_fraction <= 1:
            raise ValueError('--gpu-memory-fraction requires CUDA and a fraction in (0, 1]')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device.index)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((out / name).exists() for name in ('ckpt.pt', 'best.pt', 'log.jsonl')):
        raise ValueError('SFT output contains a run; use --resume or a new --out')
    if args.resume and not (out / 'ckpt.pt').exists():
        raise FileNotFoundError('--resume requires an existing SFT checkpoint')

    # 数据
    from tools.prepare import ensure_tokens

    args.data = ensure_tokens(args.data, chat=True, seq_len=args.seq_len)
    train_data, val_data = open_chat_data(args.data)
    seq_len = train_data.row_len - 1
    steps = total_steps(args, train_data)

    # 模型和优化器
    model = load_model(Path(args.init), device)
    model.gradient_checkpointing = device.type == "cuda" and not args.no_checkpoint
    model.checkpoint_policy = "selective"
    if model.cfg.vocab_size != train_data.vocab_size:
        raise ValueError(f"模型的词表是 {model.cfg.vocab_size}，数据是用 {train_data.vocab_size} 的词表切的")
    if seq_len > model.cfg.max_seq_len:
        raise ValueError(f"数据是按片段长度 {seq_len} 打包的，超过了模型的上限 {model.cfg.max_seq_len}")
    optimizers = optimizers_from_args(model, args)
    start_step = resume(out / "ckpt.pt", model, optimizers, device) if args.resume else 0
    learned, best_val, best_step = 0, float('inf'), 0
    if args.resume:
        state = load_checkpoint(out / 'ckpt.pt', torch.device('cpu'), mmap=True)
        previous = state.get('sft_config', {})
        for key in ('data', 'batch_size', 'grad_accum', 'seed'):
            if previous.get(key) != getattr(args, key):
                raise ValueError(f'SFT resume changes data order: {key}')
        progress = state.get('sft_progress', {})
        learned, best_val, best_step = progress.get('answer_tokens', 0), progress.get('best_val', float('inf')), progress.get('best_step', 0)
        restore_rng(state.get('rng'))
        del state
    (out / "config.json").write_text(json.dumps({"sft": vars(args), "steps": steps, "model": json.loads(model.cfg.to_json()), "data": train_data.meta},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    forward = compile_model(model, device, enabled=not args.no_compile, seq_len=seq_len,
                            attention_backend=args.attention_backend, autotune=args.autotune)
    graph_step = None
    if args.cuda_graph:
        if device.type != 'cuda' or model.cfg.moe_experts:
            raise ValueError('--cuda-graph requires CUDA and a dense model')
        from .cuda_graph import OptimizerGraph
        graph_step = OptimizerGraph(optimizers)
    print(f"设备 {device}；从 {args.init} 出发；{train_data.rows} 行对话（每行 {seq_len} 词）；"
          f"每步 {args.batch_size * args.grad_accum} 行，共 {steps} 步（约 {steps * args.batch_size * args.grad_accum / train_data.rows:.2f} 轮）", flush=True)

    # 训练循环
    model.train()
    smooth, val_loss, loss_value = None, None, float("nan")
    last_time, last_step = time.perf_counter(), start_step
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        if args.eval_before and not args.resume and val_data:
            val_loss, saturated = evaluate(model, val_data, args, device)
            best_val = val_loss
            save_checkpoint(out / 'best.pt', model, [], 0, sft_config=vars(args), validation_loss=val_loss)
            log.write(json.dumps(dict(step=0, val_loss=val_loss, saturated=saturated)) + '\n'); log.flush()
            print(f'BASELINE validation_loss={val_loss:.6f}', flush=True)
            last_time = time.perf_counter()
        for step in range(start_step, steps):
            scale = lr_scale(step, steps, args.warmup_steps, args.final_lr_frac)
            set_lr_scale(optimizers, scale * args.lr_mult)
            loss_total = None
            for micro in range(args.grad_accum):
                x, y = train_data.ordered_batch(args.batch_size, seq_len, index=step * args.grad_accum + micro, seed=args.seed, device=device)
                learned += int((y != -1).sum())
                with autocast(device):
                    loss = forward(x, y, return_logits=False).loss / args.grad_accum
                loss.backward()
                loss_total = loss.detach() if loss_total is None else loss_total + loss.detach()
            loss_value = loss_total.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"第 {step} 步的损失变成了 {loss_value}，训练停下。上一个存档在 {out / 'ckpt.pt'}")
            grad_norm = clip_gradients(model.parameters(), args.clip_grad)
            if graph_step is None:
                step_optimizers(optimizers)
            else:
                graph_step()
            smooth = loss_value if smooth is None else 0.9 * smooth + 0.1 * loss_value

            # 记录、验证、存档
            done = step + 1
            record = None
            if done % args.log_every == 0 or done == steps:
                now = time.perf_counter()
                epoch = done * args.batch_size * args.grad_accum / train_data.rows
                record = dict(step=done, epoch=round(epoch, 4), answer_tokens=learned, loss=round(loss_value, 4), smooth_loss=round(smooth, 4),
                              lr_scale=round(scale, 4),lr_mult=args.lr_mult, seconds=round(now - last_time, 2))
                if grad_norm is not None:
                    record["grad_norm"] = round(float(grad_norm), 4)
                eta = format_eta((steps - done) * (now - last_time) / max(1, done - last_step))
                last_time, last_step = now, done
                print(f"第 {done:>6} 步（{epoch:.3f} 轮，回答 {count(learned)} 词）  损失 {loss_value:.4f}（平滑 {smooth:.4f}）  学习率×{scale:.3f}"
                      + (f"  梯度 {float(grad_norm):.2f}" if grad_norm is not None else "") + f"  剩余 {eta}", flush=True)
            if val_data and (done % args.eval_every == 0 or done == steps):
                val_loss, saturated = evaluate(forward, val_data, args, device)
                record = {**(record or dict(step=done)), "val_loss": round(val_loss, 4), "saturated": round(saturated, 6)}
                print(f"第 {done:>6} 步  验证集损失 {val_loss:.4f}  塌缩 {saturated:.2%}", flush=True)
                if val_loss < best_val:
                    best_val, best_step = val_loss, done
                    save_checkpoint(out / 'best.pt', model, [], done, sft_config=vars(args), validation_loss=val_loss)
                last_time = time.perf_counter()  # 验证花的时间不算进剩余时间
            if record:
                log.write(json.dumps(record) + "\n")
                log.flush()
            if done % args.save_every == 0 or done == steps:
                save_checkpoint(out / "ckpt.pt", model, optimizers, done, sft_config=vars(args), rng=rng_state(),
                                sft_progress=dict(answer_tokens=learned, best_val=best_val, best_step=best_step))
    summary = dict(step=steps, loss=loss_value, val_loss=val_loss, tokens=learned, best_val=best_val, best_step=best_step)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SFT：只在回答上算损失")
    g = p.add_argument_group("数据和输出")
    g.add_argument("--run", default="out/small", help="运行目录，各个阶段都放在这里")
    g.add_argument("--init", default=None, help="从这个存档出发；默认是运行目录里的退火存档，没有就用预训练的")
    g.add_argument("--data", default="data/sft_small.parquet", help="对话数据：parquet（第一次用时自动打包）或 tools.prepare --chat 打包好的目录")
    g.add_argument("--seq-len", type=int, default=2048, help="--data 是 parquet 时，打包成多长的片段")
    g.add_argument("--out", default=None, help="输出目录；默认 <运行目录>/sft，专家数据（sft_expert_<领域>）是 <运行目录>/expert_<领域>")
    g.add_argument("--batch-size", type=int, default=16, help="每个批次多少行")

    g = p.add_argument_group("训练")
    g.add_argument("--epochs", type=float, default=1.0, help="把数据过几遍（没给 --steps 时用）")
    g.add_argument("--steps", type=int, default=0, help="一共更新多少次；给了就不看 --epochs")
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--warmup-steps", type=int, default=50)
    g.add_argument("--final-lr-frac", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0)
    add_optimizer_args(p, lr_mult=0.1)

    g = p.add_argument_group("记录和存档")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--eval-every", type=int, default=100)
    g.add_argument("--eval-batches", type=int, default=20)
    g.add_argument('--eval-batch-size', type=int, default=0)
    g.add_argument('--eval-micro-batch-size', type=int, default=0)
    g.add_argument('--eval-before', action='store_true')
    g.add_argument("--save-every", type=int, default=200)
    g.add_argument("--resume", action="store_true")

    g = p.add_argument_group("硬件")
    g.add_argument("--device", default="auto")
    g.add_argument("--no-compile", action="store_true")
    g.add_argument('--no-checkpoint', action='store_true')
    g.add_argument('--attention-backend', choices=('flex', 'cudnn'), default='flex')
    g.add_argument('--autotune', action='store_true')
    g.add_argument('--cuda-graph', action='store_true')
    g.add_argument('--gpu-memory-fraction', type=float, default=0.)
    args = p.parse_args(argv)
    if args.out is None:
        name = Path(args.data).stem.split("-")[0]
        args.out = str(Path(args.run) / (name.removeprefix("sft_") if name.startswith("sft_expert_") else "sft"))
    return args


def resolve_init(args) -> None:
    """没写 --init 时，在运行目录里找出发的存档（退火优先，其次预训练）。放在训练开始时做，只看参数的地方不用存档存在。"""
    if not args.init:
        args.init = str(find_stage(args.run, ("anneal", "pretrain")))


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
