"""Dataset v3 = dataset v2, every file unchanged, + the approved public Arabic books of batch 2.

Usage (from the project root):
    python -m data.books_candidate --raw data/raw/private_books/books_ar_batch2 ...   # cleaned book text first
    python -m data.add_books

Why a copy instead of re-running data.prepare: the experiment compares v2 against v2 + books, so every
v2 file (train text and all evaluation sets A, B, books, C) must stay byte-identical. This is checked.

For each book in data.sources.BOOKS_BATCH2:
  - the cleaned text must come from the EPUB with the pinned sha256 (read from the candidate provenance)
  - lines already present anywhere in v2 are dropped (exact line dedup, as in data.prepare)
  - blocks of ~2000 characters go to train / val / test (90 / 5 / 5), unless held_out is False
Sampling: the Arabic total stays the same as in v2. A book with weight "match:<source>" is seen as often
per character as that source; a number is a fixed weight. The v2 Arabic sources are scaled down together.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from data.prepare import SPLITS, assign_splits, fingerprint, make_blocks, shared_window_rate, unigram_loss
from data.sources import BOOKS_BATCH2

ROOT = Path(__file__).resolve().parents[1]


def lines_set(text, min_words=3):
    return {" ".join(l.split()) for l in text.split("\n") if len(l.split()) >= min_words}


def book_weights(v2_weights, languages, train_chars, books):
    """Sampling weights with the Arabic total of v2 kept. Returns (weights, scale applied to the v2 Arabic sources)."""
    arabic_v2 = [n for n, lang in languages.items() if lang == "ar" and n in v2_weights]
    arabic_total = sum(v2_weights[n] for n in arabic_v2)
    fixed = sum(b["weight"] for b in books if not isinstance(b["weight"], str))
    # matched books: weight = scale * v2_weight[ref] / chars[ref] * chars[book]
    per_char_matched = sum(v2_weights[b["weight"].split(":", 1)[1]] / train_chars[b["weight"].split(":", 1)[1]]
                           * train_chars[b["name"]] for b in books if isinstance(b["weight"], str))
    scale = (arabic_total - fixed) / (arabic_total + per_char_matched)
    weights = {n: (w * scale if n in arabic_v2 else w) for n, w in v2_weights.items()}
    for b in books:
        if isinstance(b["weight"], str):
            ref = b["weight"].split(":", 1)[1]
            weights[b["name"]] = scale * v2_weights[ref] / train_chars[ref] * train_chars[b["name"]]
        else:
            weights[b["name"]] = b["weight"]
    return weights, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/processed/hishamlm_v2")
    parser.add_argument("--candidate", default="data/processed/private_books_v2_candidate")
    parser.add_argument("--out", default="data/processed/hishamlm_v3")
    args = parser.parse_args()
    base, out, cand = ROOT / args.base, ROOT / args.out, ROOT / args.candidate
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    cfg = meta["config"]

    # ---- 1. exact copy of v2 ----
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("tokenizers"))
    base_files = {p.relative_to(base): p.read_bytes() for p in base.glob("*/*.txt")}

    # ---- 2. the books ----
    provenance = {p["id"]: p for p in json.loads((cand / "provenance.json").read_text(encoding="utf-8"))}
    v2_text = "\n".join(t.decode("utf-8") for t in base_files.values())
    seen = lines_set(v2_text)
    report, train_chars = {}, {n: s["train_chars"] for n, s in meta["sources"].items()}
    languages = {n: s["language"] for n, s in meta["sources"].items()}
    for book in BOOKS_BATCH2:
        name = book["name"]
        prov = provenance[name]
        if prov["sha256"] != book["sha256"]:
            raise ValueError(f"{name}: candidate text was extracted from a different file than the pinned EPUB")
        text = (cand / "books" / f"{name}.txt").read_text(encoding="utf-8")
        candidate_chars = len(text)
        kept, dropped = [], 0
        for line in text.split("\n"):
            if " ".join(line.split()) in seen and len(line.split()) >= 3:
                dropped += 1
                continue
            kept.append(line)
        text = "\n".join(kept)
        blocks = [(f"{name}#{i}", b) for i, b in enumerate(make_blocks(text, cfg["block_chars"]))]
        if book["held_out"]:
            assignment = assign_splits([(u, len(b)) for u, b in blocks], cfg["general_val"], cfg["general_test"],
                                       f"{cfg['seed']}:{name}")
        else:
            assignment = {u: "train" for u, _ in blocks}
        split_text = {s: "\n\n".join(b for u, b in blocks if assignment[u] == s) for s in SPLITS}
        for s in SPLITS:
            (out / s / f"{name}.txt").write_text(split_text[s], encoding="utf-8")
        train_chars[name] = len(split_text["train"])
        languages[name] = book["language"]
        report[name] = {"language": book["language"], "eval_group": book["eval_group"], "private": book["private"],
                        "title": book["title"], "author": book["author"], "license": book["license"],
                        "source_file": book["file"], "sha256": book["sha256"], "url": book["url"],
                        "chars_from_candidate": candidate_chars,
                        "lines_already_in_v2_dropped": dropped, "held_out": book["held_out"],
                        **{f"{s}_chars": len(split_text[s]) for s in SPLITS}}
        print(f"{name:22s} train {train_chars[name]:>8,}  val {len(split_text['val']):>7,}  test {len(split_text['test']):>7,}"
              f"  (dropped {dropped} lines already in v2)")

    # ---- 3. v2 files must be unchanged ----
    changed = [str(p) for p, data in base_files.items() if (out / p).read_bytes() != data]
    if changed:
        raise RuntimeError(f"v2 files changed: {changed}")

    # ---- 4. leakage checks ----
    train_all = "\n".join((out / "train" / p.name).read_text(encoding="utf-8") for p in sorted((out / "train").glob("*.txt")))
    train_lines = lines_set(train_all)
    new_train = "\n".join((out / "train" / f"{b['name']}.txt").read_text(encoding="utf-8") for b in BOOKS_BATCH2)
    new_names = {b["name"] for b in BOOKS_BATCH2}
    held = {s: "\n".join((out / s / p.name).read_text(encoding="utf-8") for p in sorted((out / s).glob("*.txt"))
                         if p.stem not in new_names) for s in ("val", "test")}  # the v2 evaluation sets only
    new_held = {b["name"]: "\n".join((out / s / f"{b['name']}.txt").read_text(encoding="utf-8") for s in ("val", "test"))
                for b in BOOKS_BATCH2 if b["held_out"]}
    questions = [json.loads(l)["question"] for l in (ROOT / cfg["eval_questions"]).read_text(encoding="utf-8").splitlines() if l.strip()]
    leakage = {
        # inside one book, repeated formulas and quoted verses also appear here (as for the v2 general sources)
        "new_held_out_lines_in_train": {n: len(lines_set(t) & train_lines) for n, t in new_held.items()},
        "new_held_out_shared_50char_windows_with_train": {n: round(shared_window_rate(t, train_all), 5) for n, t in new_held.items()},
        "v2_held_out_lines_in_new_train": len(lines_set(held["val"] + "\n" + held["test"]) & lines_set(new_train)),
        "v2_held_out_shared_50char_windows_with_new_train": round(shared_window_rate(held["val"] + held["test"], new_train), 5),
        "held_out_questions_in_new_train": sum(q in new_train for q in questions),
    }
    print("leakage:", leakage)

    # ---- 5. weights, eval sets, meta ----
    v2_weights = meta["sampling"]["weights_language_balanced"]
    weights, scale = book_weights(v2_weights, languages, train_chars, BOOKS_BATCH2)
    vocab_size = len(json.loads((out / "vocab.json").read_text(encoding="utf-8"))["chars"])
    train_counts = Counter(train_all)
    for b in BOOKS_BATCH2:
        if b["held_out"]:
            for s in ("val", "test"):
                text = (out / s / f"{b['name']}.txt").read_text(encoding="utf-8")
                meta["eval_sets"][f"A_{b['eval_group']}_{s}"] = {
                    "chars": len(text), "unigram_baseline_loss": round(unigram_loss(train_counts, text, vocab_size), 4)}
    meta["sources"].update(report)
    meta["sampling"]["weights_language_balanced_v2"] = v2_weights
    meta["sampling"]["weights_language_balanced"] = {n: round(w, 6) for n, w in weights.items()}
    meta["sampling"]["books_batch2_note"] = (f"Arabic total kept at {sum(v2_weights[n] for n in v2_weights if languages.get(n) == 'ar'):.4f}; "
                                             f"v2 Arabic sources scaled by {scale:.4f}")
    meta["leakage_checks_books_batch2"] = leakage
    meta["derived_from"] = {"dataset": args.base, "fingerprint": meta["dataset_fingerprint"],
                            "unchanged_files": len(base_files)}
    meta["totals"]["train_chars"] = sum(len((out / "train" / p.name).read_text(encoding="utf-8")) for p in (out / "train").glob("*.txt"))
    meta["dataset_fingerprint"] = fingerprint(out)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("weights:", {n: round(w, 4) for n, w in weights.items()}, "| Arabic total",
          round(sum(w for n, w in weights.items() if languages[n] == "ar"), 4), "| scale", round(scale, 4))
    print(f"fingerprint {meta['dataset_fingerprint']}")


if __name__ == "__main__":
    main()
