"""Evaluate a checkpoint on train, validation and test: loss and perplexity.

Usage (from the project root):
    python -m training.evaluate --checkpoint checkpoints/transformer_corpus/best.pt

Unlike the quick estimate during training (random windows), this scores EVERY
character of each validation/test set exactly once, using sliding windows:
each window has context_length tokens and only its last `stride` targets are
scored, so every prediction sees at least context_length - stride characters
of context (except at the very start of a set).
Train sets are large, so only the first --max-train-tokens of each source are scored.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from inference.generate import load_checkpoint
from training.config import TrainConfig
from training.train import ROOT, load_data
from training.utils import get_device


def perplexity(loss):
    """exp(loss), guarded against overflow for very large losses."""
    return math.exp(loss) if loss < 50 else float("inf")


@torch.no_grad()
def stream_loss(model, ids, context_length, stride, batch_size=64):
    """Mean next-token loss over a whole 1-D stream of token ids. Returns (loss, number of targets)."""
    T, n = context_length, len(ids)
    if n < 2:
        return float("nan"), 0
    if n - 1 <= T:  # a single short window
        logits = model(ids[None, :-1])
        return F.cross_entropy(logits[0], ids[1:]).item(), n - 1

    # jobs: (window start, index inside the window of the first target to score)
    # target t means "predict ids[t + 1] from ids[: t + 1]"; targets 0 .. n-2 exist.
    jobs, scored_until = [(0, 0)], T
    while scored_until < n - 1:
        end = min(scored_until + stride, n - 1)
        start = end - T
        jobs.append((start, scored_until - start))
        scored_until = end

    total, count = 0.0, 0
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]
        x = torch.stack([ids[s : s + T] for s, _ in chunk])
        y = torch.stack([ids[s + 1 : s + T + 1] for s, _ in chunk])
        logits = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(len(chunk), T)
        for row, (_, first) in enumerate(chunk):
            total += nll[row, first:].sum().item()
            count += T - first
    return total / count, count


def evaluate_checkpoint(path, max_train_tokens=200_000, stride=None, device="cpu"):
    model, tokenizer, ckpt = load_checkpoint(path, device)
    cfg = TrainConfig.from_dict(ckpt["config"])
    data = load_data(cfg, tokenizer)
    stride = stride or cfg.context_length // 2
    results = {}

    train_losses = {}
    for name, stream in data.train.items():
        loss, count = stream_loss(model, stream[:max_train_tokens].to(device), cfg.context_length, stride)
        train_losses[name] = loss
        results[f"train/{name}"] = {"loss": loss, "perplexity": perplexity(loss), "tokens": count}
    # train loss as the model saw it: sources mixed with the sampling weights
    mixed = sum(w * train_losses[n] for n, w in zip(data.names, data.weights.tolist()))
    results["train (sampling mix)"] = {"loss": mixed, "perplexity": perplexity(mixed)}

    baselines = data.info.get("baselines", {})
    for name, stream in data.evals.items():
        loss, count = stream_loss(model, stream.to(device), cfg.context_length, stride)
        results[name] = {"loss": loss, "perplexity": perplexity(loss), "tokens": count,
                         "unigram_baseline_loss": baselines.get(name, {}).get("unigram_baseline_loss")}
    return {
        "checkpoint": str(path),
        "step": ckpt["step"],
        "dataset_fingerprint": data.info.get("fingerprint"),
        "uniform_baseline_loss": math.log(tokenizer.vocab_size),
        "stride": stride,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-train-tokens", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    report = evaluate_checkpoint(args.checkpoint, args.max_train_tokens, device=get_device(args.device))
    print(f"checkpoint {report['checkpoint']} (step {report['step']}) | uniform baseline {report['uniform_baseline_loss']:.4f}")
    print(f"{'set':32s} {'loss':>8s} {'ppl':>9s} {'unigram':>8s} {'tokens':>9s}")
    for name, r in report["results"].items():
        uni = r.get("unigram_baseline_loss")
        print(f"{name:32s} {r['loss']:8.4f} {r['perplexity']:9.2f} {uni if uni is not None else '':>8} {r.get('tokens', ''):>9}")
    out = Path(args.checkpoint).with_name(f"eval_{Path(args.checkpoint).stem}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
