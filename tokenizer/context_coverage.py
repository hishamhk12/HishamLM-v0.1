"""How much text fits in a context window of N tokens?

Usage (from the project root):
    python -m tokenizer.context_coverage --tokenizer data/processed/hishamlm_v1/tokenizers/bpe_4096.json \\
                                         --dataset data/processed/hishamlm_v2 --contexts 64 128

For every group of sources (Arabic general, English, After-Sales, books) the train text is cut
into consecutive windows of N tokens. Per window we count the original characters, the words
and the sentence ends it covers. We also report how long sentences are in tokens and which
share of them fits completely inside one window.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from tokenizer import load_tokenizer

GROUPS = {
    "arabic_general": ["common_voice_ar", "ntrex_ar", "xquad_ar"],
    "english": ["common_voice_en", "gutenberg_en"],
    "after_sales": ["after_sales"],
    "books": ["hindawi_books_ar"],
}
WORD = re.compile("[^\\W\\d_](?:[^\\W\\d_]|[\u064B-\u0652\u0670\u0640])*")  # letters (Arabic or Latin) with their diacritics
SENTENCE_END = re.compile(r"[.!?؟]+(?=\s|$)|\n+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n+")


def window_stats(tokenizer, ids, n):
    rows = []
    for start in range(0, len(ids) - n + 1, n):
        text = tokenizer.decode(ids[start : start + n])
        rows.append((len(text), len(WORD.findall(text)), len(SENTENCE_END.findall(text))))
    mean = lambda k: round(statistics.mean(r[k] for r in rows), 1)
    return {"windows": len(rows), "chars": mean(0), "words": mean(1), "sentence_ends": mean(2)}


def sentence_lengths(tokenizer, text, max_sentences=3000):
    sentences = [s for s in SENTENCE_SPLIT.split(text) if WORD.search(s)][:max_sentences]
    return [len(tokenizer.encode(" " + s.strip())) for s in sentences]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--max-chars", type=int, default=300_000, help="per source, from the start of its train split")
    parser.add_argument("--out", help="optional JSON output path")
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)
    root = Path(args.dataset)
    report = {}
    for group, names in GROUPS.items():
        texts = [(root / "train" / f"{n}.txt").read_text(encoding="utf-8")[: args.max_chars] for n in names]
        ids = [i for t in texts for i in tok.encode(t)]
        lengths = sentence_lengths(tok, "\n".join(texts))
        report[group] = {
            "chars_per_token": round(sum(map(len, texts)) / len(ids), 3),
            "sentence_tokens_median": statistics.median(lengths),
            "sentence_tokens_mean": round(statistics.mean(lengths), 1),
            **{f"ctx{n}": window_stats(tok, ids, n) | {"sentences_fitting": round(sum(l <= n for l in lengths) / len(lengths), 3)}
               for n in args.contexts},
        }

    print(f"{'group':15s} {'chars/tok':>9s} {'sent. tokens (median/mean)':>27s}  " +
          "  ".join(f"{'ctx' + str(n) + ': chars / words / sentence ends / sentences that fit':>52s}" for n in args.contexts))
    for group, r in report.items():
        cols = "  ".join(f"{r[f'ctx{n}']['chars']:>14} / {r[f'ctx{n}']['words']:>5} / {r[f'ctx{n}']['sentence_ends']:>5} / "
                         f"{100 * r[f'ctx{n}']['sentences_fitting']:>5.1f}%{'':>14s}" for n in args.contexts)
        print(f"{group:15s} {r['chars_per_token']:>9.3f} {r['sentence_tokens_median']:>14} / {r['sentence_tokens_mean']:<10}  {cols}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
