"""Qualitative generalization check with fixed prompts, plus a memorization check.

Usage (from the project root):
    python -m inference.sample_eval --checkpoint checkpoints/transformer_corpus/best.pt

For every prompt in configs/eval_prompts.json:
  1. confirm the prompt does not appear verbatim in the training text
  2. generate a continuation (fixed seed)
  3. measure how much of the continuation is copied from the training text
     (longest verbatim match, share of 32-character windows found in train)
As a reference, the same measures are computed for random snippets of held-out
text: real unseen text also shares some phrases with the training corpus.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from inference.generate import generate, load_checkpoint
from inference.memorization import is_copying, memorization_report
from training.config import TrainConfig
from training.train import ROOT


def generate_chars(model, tokenizer, prompt, n_chars, temperature, top_k, generator):
    """Continue `prompt` until the continuation has n_chars characters (the same length for every tokenizer)."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    n_prompt = ids.size(1)
    while len(tokenizer.decode(ids[0, n_prompt:].tolist())) < n_chars:
        ids = generate(model, ids, 1, temperature, top_k, generator)
    return tokenizer.decode(ids[0, n_prompt:].tolist())[:n_chars]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", default="configs/eval_prompts.json")
    parser.add_argument("--max-new-chars", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-snippets", type=int, default=20)
    args = parser.parse_args()

    model, tokenizer, ckpt = load_checkpoint(args.checkpoint, "cpu")
    cfg = TrainConfig.from_dict(ckpt["config"])
    if not cfg.dataset:
        raise ValueError("sample_eval needs a checkpoint trained on a prepared dataset")
    dataset = ROOT / cfg.dataset
    train_text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((dataset / "train").glob("*.txt")))
    held_out = "\n".join(p.read_text(encoding="utf-8") for split in ("val", "test")
                         for p in sorted((dataset / split).glob("*.txt")))

    # reference: how much does genuinely unseen text overlap with train?
    rng = random.Random(args.seed)
    refs = []
    for _ in range(args.reference_snippets):
        start = rng.randrange(len(held_out) - args.max_new_chars)
        refs.append(memorization_report(held_out[start : start + args.max_new_chars], train_text))
    reference = {
        "longest_match_median": statistics.median(r["longest_match"] for r in refs),
        "longest_match_max": max(r["longest_match"] for r in refs),
        "overlap_32gram_median": statistics.median(r["overlap_32gram"] for r in refs),
    }
    print(f"reference (held-out text): {reference}")

    prompts = json.loads((ROOT / args.prompts).read_text(encoding="utf-8"))
    results = []
    for group, items in prompts.items():
        for prompt in items:
            gen = torch.Generator().manual_seed(args.seed + len(results))  # a different seed for every prompt
            continuation = generate_chars(model, tokenizer, prompt, args.max_new_chars, args.temperature, args.top_k, gen)
            report = memorization_report(continuation, train_text)
            result = {"group": group, "prompt": prompt, "prompt_in_train": prompt in train_text,
                      "continuation": continuation, **report, "copying_warning": is_copying(report)}
            results.append(result)
            flag = "  WARNING: looks copied from training text" if result["copying_warning"] else ""
            seen = "  (prompt appears in train!)" if result["prompt_in_train"] else ""
            print(f"\n[{group}] {prompt}{seen}\n{prompt}{continuation}")
            print(f"  longest verbatim match {report['longest_match']} chars | 32-gram overlap {report['overlap_32gram']:.2f}{flag}")

    out = Path(args.checkpoint).with_name(f"samples_{Path(args.checkpoint).stem}.json")
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "step": ckpt["step"], "settings": vars(args),
                               "reference": reference, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
