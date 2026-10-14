"""Does generated Arabic drift toward the vocabulary of one training source?

Usage (from the project root):
    python -m inference.vocab_drift --samples A=checkpoints/X/samples_best_long_ar.json B=... --out drift.json

Two word lists are taken from the training text itself (no hand-made lists):
  hindawi    words much more frequent in the Hindawi books than in general Arabic (computing, data, machines, theatre ...)
  new_books  words much more frequent in the new books (Kalila wa Dimna, al-Bayan) than in general Arabic and Hindawi
A word is "distinctive" when it occurs at least --min-count times in the target and its frequency per
million words is at least --ratio times higher than in the background (add-one smoothed).
For each samples file: share of Arabic words (light prefix stripping) that belong to each list, per 100 words.
The same rate on real held-out general Arabic is the reference.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from inference.coherence import normalize, stem, words

ROOT = Path(__file__).resolve().parents[1]
ARABIC = re.compile("[\u0621-\u064A]")
GENERAL = ["common_voice_ar", "ntrex_ar", "xquad_ar"]


def counts(dataset, names, split="train"):
    c = Counter()
    for n in names:
        path = dataset / split / f"{n}.txt"
        if path.exists():
            c.update(stem(w) for w in words(path.read_text(encoding="utf-8")) if len(w) >= 3 and ARABIC.match(w))
    return c


def distinctive(target, background, min_count, ratio, top):
    t_total, b_total = sum(target.values()), sum(background.values())
    scored = []
    for w, n in target.items():
        if n < min_count:
            continue
        r = (n / t_total) / ((background.get(w, 0) + 1) / b_total)
        if r >= ratio:
            scored.append((r, w))
    return [w for _, w in sorted(scored, reverse=True)[:top]]


def rate(text, vocab):
    ws = [stem(w) for w in words(normalize(text)) if len(w) >= 3 and ARABIC.match(w)]
    return 100 * sum(w in vocab for w in ws) / max(len(ws), 1)


def load_texts(path, groups):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "results" in data:  # inference.sample_eval
        return [r["continuation"] for r in data["results"]]
    label = next(iter(data["models"]))  # inference.coherence: label given as path#label
    return [s["continuation"] for s in data["models"][label]["samples"] if s["group"] in groups]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True, help="label=path.json or label=coherence.json#model")
    parser.add_argument("--dataset", default="data/processed/hishamlm_v3")
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--ratio", type=float, default=8.0)
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--groups", default="arabic", help="coherence groups to use (comma separated)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    general, hindawi = counts(dataset, GENERAL), counts(dataset, ["hindawi_books_ar"])
    new_books = counts(dataset, ["kalila_wa_dimna", "al_bayan_wal_tabyin", "nazarat_al_bakhil"])
    lists = {"hindawi": set(distinctive(hindawi, general, args.min_count, args.ratio, args.top)),
             "new_books": set(distinctive(new_books, general + hindawi, args.min_count, args.ratio, args.top))}
    report = {"settings": vars(args), "lists": {k: sorted(v) for k, v in lists.items()}, "results": {}}
    print({k: len(v) for k, v in lists.items()}, "| examples:", {k: sorted(v)[:12] for k, v in lists.items()})

    groups = set(args.groups.split(","))
    for spec in args.samples:
        label, path = spec.split("=", 1)
        path, _, model = path.partition("#")
        if model:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            texts = [s["continuation"] for s in data["models"][model]["samples"] if s["group"] in groups]
        else:
            texts = load_texts(path, groups)
        report["results"][label] = {k: round(sum(rate(t, v) for t in texts) / len(texts), 3) for k, v in lists.items()} \
            | {"samples": len(texts), "share_with_any_hindawi_word": round(sum(rate(t, lists["hindawi"]) > 0 for t in texts) / len(texts), 3)}

    ref = "\n".join((dataset / s / f"{n}.txt").read_text(encoding="utf-8") for s in ("test",) for n in GENERAL)
    chunks = [ref[i:i + 400] for i in range(0, len(ref) - 400, 400)][:200]
    report["results"]["reference: real held-out general Arabic"] = {
        k: round(sum(rate(t, v) for t in chunks) / len(chunks), 3) for k, v in lists.items()} | {"samples": len(chunks)}
    for label, r in report["results"].items():
        print(f"{label:45s} hindawi words /100: {r['hindawi']:6.2f}   new-books words /100: {r['new_books']:6.2f}   (n={r['samples']})")
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
