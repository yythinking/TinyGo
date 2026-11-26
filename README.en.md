# TinyGo

[![License: MIT](https://img.shields.io/badge/license-MIT-2140d6)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-2140d6)](pyproject.toml)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-e9482b)](https://pytorch.org)

[中文](README.md)

TinyGo is a Chinese-first small language model project built with PyTorch. It includes a tokenizer, data preparation, training, evaluation, and local inference.

- **Model.** Multi-stream residuals (mHC), Engram lookup memory, 3:1 sliding-window attention, and optional mixture of experts (MoE).
- **Data.** Scripts for downloading, cleaning, deduplicating, and grouping public data into tiers.
- **Training.** Pretraining, annealing, SFT, domain experts, multi-teacher on-policy distillation (MOPD), and reinforcement learning (RLVR, agent RL, and RLAIF).
- **Chinese.** The tokenizer, data mixture, and grading rubrics are designed for Chinese, including classical Chinese and poetry.

## Quick start

Requires Python 3.11+ and PyTorch 2.5+. The installed command is `tinybrain`.

```bash
git clone https://github.com/yythinking/TinyGo.git && cd TinyGo
pip install -e ".[artifact]"
```

### Local chat

Download the Small Chat weights from [the resource index](README.md#来源与致谢), then start a terminal chat or a web server:

```bash
tinybrain chat out/models/small-chat
tinybrain serve out/models/small-chat
```

Open `http://127.0.0.1:8001/` for the web chat. Add `--demo` to `chat` to try the built-in prompts.

The `chat`, `serve`, `generate`, and `eval` commands accept a model directory and verify its files against the SHA-256 hashes in `artifact.json`. Use `tinybrain export --checkpoint <ckpt.pt> --out <dir> --stage <stage>` to export your own checkpoint in the same format.

### Train from scratch

Place `pretrain_nano.parquet` and `sft_small.parquet` under `data/`. Download them from [the resource index](README.md#来源与致谢) or build the data with `tinybrain data` (`tinybrain data --check` lists the required inputs).

```bash
pip install -e ".[data]"
# Optional: pip install -e ".[cuda,fast]"

tinybrain pretrain     # data/pretrain_nano.parquet → out/small/pretrain
tinybrain sft          # continues from the previous stage → out/small/sft
tinybrain generate     # continue a few built-in prompts
tinybrain chat         # chat (--tools enables a calculator and Python)
tinybrain serve        # chat page at http://127.0.0.1:8001/ (continue mode for base checkpoints), plus an OpenAI-compatible API
```

Each stage writes to `out/<preset>/<stage>/` and starts from the previous stage's checkpoint, so no paths are needed. Use `--preset medium` for a bigger model and `--data data/pretrain_mini.parquet` for more data. `tinybrain demo` runs every stage on the 23M demo model. `tinybrain <command> --help` lists all options.

### Full small pipeline

Pretraining → annealing → SFT → RL → evaluation. Place `pretrain_mini.parquet`, `sft_small.parquet`, `rl_small.parquet`, and `eval_tasks.parquet` under `data/` (see [the resource index](README.md#来源与致谢)). The judge model can run in a separate terminal or on another machine.

```bash
# Terminal 1
tinybrain benchmarks prepare                                         # C-Eval, CMMLU, GSM8K for local evaluation
tinybrain pretrain --data data/pretrain_mini.parquet                 # → out/small/pretrain
tinybrain prepare --chat --train-on all --input data/sft_small.parquet --out data/tokens/chat-all
tinybrain pretrain --init out/small/pretrain/ckpt.pt --data data/pretrain_mini.parquet:0.9,data/tokens/chat-all:0.1 \
    --epochs 0.1 --warmup-steps 0 --lr-mult 0.1 --final-lr-frac 0     # annealing → out/small/anneal
tinybrain sft                                                        # → out/small/sft
JUDGE_URL='http://127.0.0.1:8080/v1'  # Set this to the judge server's URL
tinybrain rl --judge-url "$JUDGE_URL" --judge-workers 1 --judge-timeout 600   # → out/small/rl
tinybrain eval out/small/rl/ckpt.pt --tasks data/eval_tasks.parquet --bench ceval cmmlu gsm8k \
    --judge-url "$JUDGE_URL" --judge-workers 1

# terminal 2: the judge (Qwen3.8-27B) does not fit in 16 GB; on an Apple-silicon Mac with 32 GB+:
pip install mlx-lm
python -m mlx_lm.server --model mlx-community/Qwen3.8-27B-4bit --host 0.0.0.0 --port 8080
```

mlx_lm.server answers one request at a time, hence `--judge-workers 1`; with vLLM on a 40 GB+ GPU you can drop it. If you already have a pretrained checkpoint, start from the annealing step with your own `--init` and data; `--steps 10000` (about 330M tokens) keeps it short.

## Model

```
token ids ─┬─ embeddings ───────────────┐
           └─ Engram lookup ─┬─ 2-gram ─┤  three "sources"
                             └─ 3-gram ─┘
                                  │ entry: each stream attends over the sources
                                  ▼
                        n = 4 residual streams
                                  │
        ┌────────── layer i (× L) ──────────────┐
        │ [layer 1, middle layer] Engram gate   │
        │ read → RMSNorm → attention → write    │
        │ read → RMSNorm → FFN / MoE → write    │
        └───────────────────────────────────────┘
                                  │ exit: attention merges the 4 streams
                                  ▼
                       RMSNorm → output head → logits
```

- **mHC.** The residual is widened to 4 streams. Each sublayer reads a weighted mix of the streams, writes its output back with learned weights, and mixes the streams with a doubly stochastic matrix (Sinkhorn, 20 log-domain iterations), so the total signal is neither amplified nor shrunk across layers. The model uses the static variant: all coefficients are input-independent and computed once. `lm/layers/hyper.py`
- **Engram.** A hashed lookup over 2- and 3-grams (8 hash heads each). The retrieved vectors are used as extra input sources and gated into the streams before layer 1 (and a middle layer for medium/large). A lookup costs memory, not compute, so the tables are large: 45% of trunk + memory parameters in small/medium. `lm/layers/engram.py`
- **Attention.** GQA, QK-norm, RoPE, context 8192; 3 of every 4 layers see only the last 1024 tokens. On CUDA this is FlexAttention with a block-sparse mask (or cuDNN with `--attention-backend cudnn`). `lm/layers/attention.py`
- **FFN / MoE.** ReLU² FFNs. The large preset keeps layer 1 dense and uses 2 shared + 48 routed quarter-width experts with top-8 routing, sqrt-softplus scores, bias-based load balancing and a 1e-4 balance loss. `lm/layers/moe.py`

| Preset | Width × layers | Trunk | Memory tables | Embeddings + head | Total |
|---|---|---|---|---|---|
| small | 768 × 10 | 64M | 53M | 50M | 167M |
| medium | 1152 × 16 | 233M | 190M | 75M | 499M |
| large | 1280 × 20 | 889M (266M active) | 207M | 84M | 1.18B |

## Tokenizer

SentencePiece BPE with 32,768 tokens, byte fallback, digit splitting, and lossless decoding.

## Dataset

The data pipeline builds four nested tiers: nano, mini, base, and full. Each Parquet file contains one tier.

Mixture for nano/mini/base: Chinese articles 37%, encyclopedias 15.5%, code 12%, math 10%, papers 8%, classical Chinese and public-domain books 6%, English web 5%, English books 4%, synthetic 2.25%, law 0.25%. full is everything left after dedup.

How it's made: per-document cleaning (mojibake, punctuation, traditional→simplified, length/language/ad/repetition filters, phone numbers, ID numbers and e-mails replaced by placeholders; court judgments excluded because they contain personal details); cross-source dedup with sentence fingerprints (2 of the 3 longest sentences must match); quality-ranked tier selection; verification of counts, duplicates, nesting and shuffling. Each row carries its source's license; there is no single license for the mixture.

## Training

- Next-token cross-entropy, computed in a form that stays correct when the model is very confident (softplus of the margin; Barron et al., arXiv 2610.31642).
- Muon for all matrices (Nesterov momentum, 5 Newton–Schulz iterations, decoupled weight decay 0.1 via `--weight-decay`), Adam for embeddings, head, memory tables and scalars.
- Cosine or WSD learning-rate schedules; WSD runs can branch a cooldown from any checkpoint.
- Epoch-based data order that is fully determined by the seed, so `--resume` gives the same result as an uninterrupted run.
- Multi-GPU with ZeRO-1: launch with `torchrun --standalone --nproc-per-node 8 -m training.train_pretrain ...`. Each parameter is owned by one GPU, which receives its averaged gradient, updates it and broadcasts it back, so optimizer state is split across GPUs. Data order depends only on the seed and the step, so with the same rows per step (`--batch-size × --grad-accum × GPUs`) a multi-GPU run trains the same model as a single-GPU run, and checkpoints resume across the two. Pretraining only for now.
- CUDA: Liger fused cross-entropy, RMSNorm, RoPE and ReLU², FlexAttention or cuDNN attention, per-layer compile, selective recompute, optional CUDA graphs for the optimizer step.

## Post-training

- **SFT** on 18 external sources plus self-made literature data (poems on a given title, classical↔modern translation, Lu Xun continuations); loss only on answers.
- **Domain experts** (math, code, literature) branched from the SFT model.
- **MOPD:** the student writes its own answers; the expert for each domain scores every position, and the student minimizes the reverse KL over the full vocabulary.
- **RL:** Dr. GRPO, fully on-policy (one update per batch), with 10 task types: math, code (sandboxed tests), logic (reasoning-gym), multiple choice, instruction following, reading (F1), poetry cloze, tool use, LLM-judge (Qwen3.8-27B), and classical poetry (tonal pattern + rhyme + judge).

Details: [docs/post-training.md](docs/post-training.md) (Chinese).

## License and data

Code: [MIT](LICENSE). Each dataset document carries its source's license. See [dataset/sources.py](dataset/sources.py) for the source list. C-Eval and CMMLU are non-commercial benchmarks downloaded locally for evaluation.

## Sources and credits

[Upstream project, downloads, and technical references](README.md#来源与致谢)
