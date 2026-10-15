"""Evaluate a checkpoint on train, validation and test: loss and perplexity.

Usage (from the project root):
    python -m training.evaluate --checkpoint checkpoints/transformer_corpus/best.pt

Unlike the quick estimate during training (random windows), this scores EVERY
character of each validation/test set exactly once, using sliding windows:
each window has context_length tokens and only its last `stride` targets are
scored, so every prediction sees at least context_length - stride characters
of context (except at the very start of a set).
Train sets are large, so only the first --max-train-chars characters of each source are scored.

Reported per set: loss per token (nats), perplexity per token, and bits per character
(BPC), the measure that stays comparable across tokenizers.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from inference.generate import load_checkpoint
from training.config import TrainConfig
from training.train import LN2, ROOT, load_data
from training.utils import get_device


def perplexity(loss):
    """exp(loss), guarded against overflow for very large losses."""
    return math.exp(loss) if loss < 50 else float("inf")


@torch.no_grad()
def stream_loss(model, ids, context_length, stride, char_len=None, batch_size=64):
    """Mean next-token loss over a whole 1-D stream of token ids.

    Returns (loss per token, number of scored targets, characters those targets stand for).
    """
    T, n = context_length, len(ids)
    char_len = char_len if char_len is not None else torch.ones(int(ids.max()) + 1)
    if n < 2:
        return float("nan"), 0, 0.0
    if n - 1 <= T:  # a single short window
        logits = model(ids[None, :-1])
        return F.cross_entropy(logits[0], ids[1:]).item(), n - 1, char_len[ids[1:]].sum().item()

    # jobs: (window start, index inside the window of the first target to score)
    # target t means "predict ids[t + 1] from ids[: t + 1]"; targets 0 .. n-2 exist.
    jobs, scored_until = [(0, 0)], T
    while scored_until < n - 1:
        end = min(scored_until + stride, n - 1)
        start = end - T
        jobs.append((start, scored_until - start))
        scored_until = end

    total, count, chars = 0.0, 0, 0.0
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]
        x = torch.stack([ids[s : s + T] for s, _ in chunk])
        y = torch.stack([ids[s + 1 : s + T + 1] for s, _ in chunk])
        logits = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(len(chunk), T)
        for row, (_, first) in enumerate(chunk):
            total += nll[row, first:].sum().item()
            count += T - first
            chars += char_len[y[row, first:]].sum().item()
    return total / count, count, chars


def bpc(loss, count, chars):
    return loss * count / (LN2 * chars)


def evaluate_checkpoint(path, max_train_chars=200_000, stride=None, device="cpu", dataset=None, context_length=None):
    """dataset: evaluate on another prepared dataset (same tokenizer) instead of the one the model was trained on.
    context_length: score with shorter windows than the model was trained with (how much does the extra context help?).
    """
    model, tokenizer, ckpt = load_checkpoint(path, device)
    cfg = TrainConfig.from_dict(ckpt["config"])
    if dataset:
        cfg.dataset = dataset
        # another dataset may not have the training run's sampling option; the evaluation sets do not depend on it
        sampling = json.loads((ROOT / dataset / "meta.json").read_text(encoding="utf-8"))["sampling"]
        if cfg.sampling not in sampling:
            cfg.sampling = "weights_language_balanced"
    if context_length:
        if context_length > cfg.context_length:
            raise ValueError(f"the model was trained with context {cfg.context_length}, cannot score {context_length}")
        cfg.context_length = context_length
    data = load_data(cfg, tokenizer)
    stride = stride or cfg.context_length // 2
    results = {}

    char_len = data.char_len.to(device)

    def score(stream):
        loss, count, chars = stream_loss(model, stream.to(device), cfg.context_length, stride, char_len)
        return {"loss": loss, "perplexity": perplexity(loss), "bpc": bpc(loss, count, chars),
                "tokens": count, "chars": int(chars)}

    for name, stream in data.train.items():
        # the same amount of text for every tokenizer: cut at max_train_chars characters
        cut = int(torch.searchsorted(torch.cumsum(data.char_len[stream], 0), max_train_chars)) + 1
        results[f"train/{name}"] = score(stream[:cut])
    # train BPC as the model saw it: sources mixed with the sampling weights
    mixed = sum(w * results[f"train/{n}"]["bpc"] for n, w in zip(data.names, data.weights.tolist()))
    results["train (sampling mix)"] = {"bpc": mixed}

    baselines = data.info.get("baselines", {})
    for name, stream in data.evals.items():
        results[name] = score(stream)
        uni = baselines.get(name, {}).get("unigram_baseline_loss")
        results[name]["unigram_baseline_bpc"] = uni / LN2 if uni is not None else None
    return {
        "checkpoint": str(path),
        "step": ckpt["step"],
        "tokenizer": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "dataset": cfg.dataset,
        "dataset_fingerprint": data.info.get("fingerprint"),
        "uniform_baseline_loss_per_token": math.log(tokenizer.vocab_size),
        "context_length": cfg.context_length,
        "stride": stride,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-train-chars", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", help="evaluate on this prepared dataset instead of the training one")
    parser.add_argument("--context-length", type=int, help="score with windows shorter than the trained context")
    parser.add_argument("--tag", default="", help="suffix for the output file name")
    args = parser.parse_args()

    report = evaluate_checkpoint(args.checkpoint, args.max_train_chars, device=get_device(args.device),
                                 dataset=args.dataset, context_length=args.context_length)
    print(f"checkpoint {report['checkpoint']} (step {report['step']}) | {report['tokenizer']} vocab {report['vocab_size']}")
    print(f"{'set':26s} {'BPC':>7s} {'loss/tok':>9s} {'ppl/tok':>8s} {'unigram BPC':>12s} {'tokens':>8s} {'chars':>8s}")
    for name, r in report["results"].items():
        uni = r.get("unigram_baseline_bpc")
        print(f"{name:26s} {r['bpc']:7.4f} {r.get('loss', float('nan')):9.4f} {r.get('perplexity', float('nan')):8.2f} "
              f"{f'{uni:.4f}' if uni else '':>12s} {r.get('tokens', ''):>8} {r.get('chars', ''):>8}")
    out = Path(args.checkpoint).with_name(f"eval_{Path(args.checkpoint).stem}{'_' + args.tag if args.tag else ''}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
