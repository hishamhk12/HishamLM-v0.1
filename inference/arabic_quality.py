"""Simple, transparent measures of generated Arabic text.

Usage (from the project root):
    python -m inference.arabic_quality --samples checkpoints/RUN/samples_best_long_ar.json [...] \\
                                       --lexicon-dataset data/processed/hishamlm_v2

For each samples file (written by inference.sample_eval):
  unknown_word_rate   share of Arabic words never seen in the lexicon (all Arabic words of the
                      lexicon dataset's TRAIN split): a proxy for invented words
  distinct_2          distinct word bigrams / all word bigrams (low = repetitive)
  words_per_sentence  average sentence length in words
The same measures on real held-out Arabic text of the same length give the reference.
"""

import argparse
import json
import random
import re
from pathlib import Path

_MARKS = re.compile(r"[ً-ْٰـ]")  # diacritics, superscript alef, tatweel
_WORD = re.compile(r"[ء-ي]+")
_SENTENCE_END = re.compile(r"[.!?؟؛\n]+")


def arabic_words(text):
    return _WORD.findall(_MARKS.sub("", text))


def measures(text, lexicon):
    words = arabic_words(text)
    bigrams = list(zip(words, words[1:]))
    sentences = [s for s in _SENTENCE_END.split(text) if arabic_words(s)]
    return {
        "arabic_words": len(words),
        "unknown_word_rate": sum(w not in lexicon for w in words) / max(len(words), 1),
        "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
        "words_per_sentence": len(words) / max(len(sentences), 1),
    }


def average(rows):
    keys = ("unknown_word_rate", "distinct_2", "words_per_sentence")
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys} | {"samples": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--lexicon-dataset", default="data/processed/hishamlm_v2")
    parser.add_argument("--reference-sources", default="hindawi_books_ar,common_voice_ar,ntrex_ar,xquad_ar")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.lexicon_dataset)
    lexicon = set(arabic_words("\n".join(p.read_text(encoding="utf-8") for p in (root / "train").glob("*.txt"))))
    print(f"lexicon: {len(lexicon):,} distinct Arabic words from {root / 'train'}")

    results = {}
    lengths = []
    for path in args.samples:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [measures(r["continuation"], lexicon) for r in data["results"]]
        lengths += [len(r["continuation"]) for r in data["results"]]
        results[path] = average(rows)

    # reference: real held-out Arabic text (test split), same length as the generations
    rng = random.Random(args.seed)
    held_out = "\n".join((root / "test" / f"{n}.txt").read_text(encoding="utf-8")
                         for n in args.reference_sources.split(",") if (root / "test" / f"{n}.txt").exists())
    n = max(lengths)
    ref = [measures(held_out[s : s + n], lexicon) for s in (rng.randrange(len(held_out) - n) for _ in range(60))]
    results["reference: real held-out Arabic"] = average(ref)

    print(f"{'':72s} {'unknown words':>13s} {'distinct-2':>10s} {'words/sentence':>14s}")
    for name, r in results.items():
        print(f"{name[-72:]:72s} {100 * r['unknown_word_rate']:12.1f}% {r['distinct_2']:10.3f} {r['words_per_sentence']:14.1f}")
    out = Path(args.samples[0]).parent / "arabic_quality.json"
    out.write_text(json.dumps({"lexicon_dataset": str(root), "lexicon_size": len(lexicon), "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
