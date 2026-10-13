"""How efficiently does a tokenizer cover each source of a prepared dataset?

Usage (from the project root):
    python -m tokenizer.coverage --tokenizer data/processed/hishamlm_v1/tokenizers/bpe_4096.json \\
                                 --dataset data/processed/hishamlm_v2

Per source and split: characters per token (higher = more compression) and how much text
falls back to raw UTF-8 byte tokens because its characters are not in the vocabulary.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from tokenizer import load_tokenizer


def coverage(tokenizer, text):
    ids = tokenizer.encode(text)
    char_len = tokenizer.token_char_lengths()
    byte_ids = [i for i in ids if i < 256]
    fallback_chars = Counter(ch for ch in text if getattr(tokenizer, "char_to_id", None) is not None
                             and ch not in tokenizer.char_to_id)
    return {
        "chars": len(text),
        "tokens": len(ids),
        "chars_per_token": round(len(text) / max(len(ids), 1), 3),
        "byte_fallback_token_share": round(len(byte_ids) / max(len(ids), 1), 6),
        "byte_fallback_char_share": round(sum(char_len[i] for i in byte_ids) / max(len(text), 1), 6),
        "fallback_characters": fallback_chars.most_common(15),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", help="optional JSON output path")
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)
    root = Path(args.dataset)
    report = {}
    print(f"{'source':18s} {'split':5s} {'chars':>10s} {'tokens':>9s} {'chars/tok':>9s} {'byte tok %':>10s} {'byte char %':>11s}  fallback characters")
    for path in sorted(root.glob("*/*.txt")):
        split, name = path.parent.name, path.stem
        r = coverage(tok, path.read_text(encoding="utf-8"))
        report[f"{name}/{split}"] = r
        chars = " ".join(f"{c}×{n}" for c, n in r["fallback_characters"][:8])
        print(f"{name:18s} {split:5s} {r['chars']:>10,} {r['tokens']:>9,} {r['chars_per_token']:>9.3f} "
              f"{100 * r['byte_fallback_token_share']:>9.3f}% {100 * r['byte_fallback_char_share']:>10.4f}%  {chars}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
