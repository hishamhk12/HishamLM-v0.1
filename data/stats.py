"""Character statistics for a text corpus.

Usage (from the project root):
    python -m data.stats                                        # data/raw/**/*.txt
    python -m data.stats --path data/processed/domain_after_sales/corpus.txt
"""

import argparse
import unicodedata
from collections import Counter
from pathlib import Path

from data.clean import clean_text


def is_arabic(ch):
    o = ord(ch)
    return 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF


def is_latin(ch):
    return ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN")


def text_stats(text, top=20):
    counts = Counter(text)
    total = len(text) or 1
    arabic = sum(n for ch, n in counts.items() if is_arabic(ch))
    latin = sum(n for ch, n in counts.items() if is_latin(ch))
    digits = sum(n for ch, n in counts.items() if ch.isdigit())
    spaces = sum(n for ch, n in counts.items() if ch.isspace())
    return {
        "characters": len(text),
        "lines": text.count("\n") + 1 if text else 0,
        "arabic_pct": 100 * arabic / total,
        "latin_pct": 100 * latin / total,
        "digit_pct": 100 * digits / total,
        "whitespace_pct": 100 * spaces / total,
        "other_pct": 100 * (total - arabic - latin - digits - spaces) / total,
        # share of Arabic among all Arabic + Latin characters
        "arabic_share_of_letters": 100 * arabic / ((arabic + latin) or 1),
        "unique_characters": len(counts),
        "vocab_size": len(counts) + 1,  # + <unk>
        "top_characters": counts.most_common(top),
        "rare_characters": sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[:top],
    }


def _show(ch):
    return {"\n": "\\n", " ": "' '", "\t": "\\t"}.get(ch, ch)


def print_stats(s, n_files=None):
    if n_files is not None:
        print(f"files:              {n_files}")
    print(f"characters:         {s['characters']:,}")
    print(f"lines:              {s['lines']:,}")
    print(f"arabic:             {s['arabic_pct']:.1f}%")
    print(f"latin:              {s['latin_pct']:.1f}%")
    print(f"digits:             {s['digit_pct']:.1f}%")
    print(f"whitespace:         {s['whitespace_pct']:.1f}%")
    print(f"other (punct, ...): {s['other_pct']:.1f}%")
    print(f"arabic / (arabic + latin): {s['arabic_share_of_letters']:.1f}%")
    print(f"unique characters:  {s['unique_characters']} (vocab size with <unk>: {s['vocab_size']})")
    print("top characters:     " + "  ".join(f"{_show(c)}:{n}" for c, n in s["top_characters"]))
    print("rare characters:    " + "  ".join(f"{_show(c)}(U+{ord(c):04X}):{n}" for c, n in s["rare_characters"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/raw", help="a .txt file or a directory searched recursively")
    parser.add_argument("--no-clean", action="store_true", help="show statistics of the raw text")
    args = parser.parse_args()

    path = Path(args.path)
    files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    texts = [f.read_text(encoding="utf-8") for f in files]
    if not args.no_clean:
        texts = [clean_text(t) for t in texts]
    print_stats(text_stats("\n".join(texts)), n_files=len(files))


if __name__ == "__main__":
    main()
