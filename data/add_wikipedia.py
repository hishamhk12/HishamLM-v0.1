"""Temporary comparison dataset: dataset v2 (every file unchanged) + a 20M-character Arabic Wikipedia sample.

Usage (from the project root), after data.wikipedia_candidate:
    python -m data.add_wikipedia

Sample (from data/processed/wikipedia_ar_candidate/articles.jsonl, never "the first 20M characters"):
  - extra filters on top of the candidate: 800 to 50,000 clean characters, templates + tables at most 60% of
    the wikitext, no mass-produced opening sentence (the same masked first sentence in >= 20 articles)
  - balanced over the ten topics: equal character budgets, a topic with too little text gives its share to
    the others, no topic above 20%; random articles inside a topic (seed 42)
  - whole articles go to train / val / test (90 / 5 / 5); held-out lines that share a 50-character run with
    Wikipedia train are dropped
Sampling: Wikipedia gets --weight of all training batches, taken only from the general Arabic sources
(Common Voice, NTREX, XQuAD); Hindawi books, English and After-Sales keep their v2 weights.
"""

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from data.clean import clean_text
from data.prepare import SPLITS, assign_splits, fingerprint, unigram_loss
from data.wikipedia_candidate import ARABIC, norm_para
from data.wikipedia_clean import mediawiki_to_text

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/processed/wikipedia_ar_work"
GENERAL_AR = ("common_voice_ar", "ntrex_ar", "xquad_ar")
NAME = "wikipedia_ar"


def template_share(line):
    row = json.loads(line)
    stats = Counter()
    mediawiki_to_text(row["wikitext"], stats)
    return row["id"], (stats["template_chars"] + stats["table_chars"]) / max(len(row["wikitext"]), 1)


def opening_skeleton(article):
    first = next((p for p in article["text"].split("\n") if len(p.split()) >= 5), "")
    s = first.replace(article["title"].split(" (")[0], "T")
    return re.sub(r"[0-9٠-٩]+", "0", norm_para(s))


def window_hashes(text, n=50):
    """Hashes of every n-character window (a set lookup is much faster than `in` on a long string)."""
    return {hash(text[i:i + n]) for i in range(len(text) - n + 1)}


def shared_window_rate(held_out, hashes, n=50, step=50):
    windows = range(0, len(held_out) - n + 1, step)
    return sum(hash(held_out[i:i + n]) in hashes for i in windows) / max(len(windows), 1)


def decontaminate(held_out, train_lines, hashes, n=50, step=10):
    """Same rule as data.prepare.decontaminate: drop held-out lines found in train or sharing an n-char run."""
    kept, dropped = [], 0
    for line in held_out.split("\n"):
        key = " ".join(line.split())
        if key and (key in train_lines or any(hash(line[i:i + n]) in hashes for i in range(0, len(line) - n + 1, step))):
            dropped += 1
            continue
        kept.append(line)
    return clean_text("\n".join(kept)), dropped


def balanced_budgets(available, total, cap_share):
    """Equal shares per topic; a topic with less text gives the rest to the others; no topic above cap."""
    budgets, open_topics, left = {}, set(available), total
    while open_topics and left > 1:
        share = min(left / len(open_topics), cap_share * total)
        short = {t for t in open_topics if available[t] - budgets.get(t, 0) <= share}
        if not short:
            for t in open_topics:
                budgets[t] = budgets.get(t, 0) + share
            break
        for t in short:
            add = available[t] - budgets.get(t, 0)
            budgets[t] = budgets.get(t, 0) + add
            left -= add
            open_topics.discard(t)
    return budgets


def add_proportional_weights(out, weight):
    """A second sampling option for the same files: Wikipedia's share is taken from every Arabic source in
    proportion to its v2 weight (each one loses the same fraction); English and After-Sales are untouched."""
    meta_path = ROOT / out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    v2_w = meta["sampling"]["weights_language_balanced_v2"]
    arabic = [n for n in v2_w if meta["sources"][n]["language"] == "ar"]
    total = sum(v2_w[n] for n in arabic)
    scale = (total - weight) / total
    weights = {n: (w * scale if n in arabic else w) for n, w in v2_w.items()} | {NAME: weight}
    meta["sampling"]["weights_language_balanced_proportional"] = {n: round(w, 6) for n, w in weights.items()}
    meta["sampling"]["proportional_note"] = (f"Wikipedia {weight:.2%} taken from all Arabic sources {arabic} in proportion "
                                             f"(each scaled by {scale:.4f}); English and After-Sales unchanged")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")  # text files untouched
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/processed/hishamlm_v2")
    parser.add_argument("--candidate", default="data/processed/wikipedia_ar_candidate/articles.jsonl")
    parser.add_argument("--out", default="data/processed/hishamlm_v2_wiki20m")
    parser.add_argument("--chars", type=int, default=20_000_000)
    parser.add_argument("--weight", type=float, default=0.10)
    parser.add_argument("--cap-share", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add-proportional-weights", action="store_true",
                        help="only add the proportional sampling option to an existing --out dataset")
    args = parser.parse_args()
    if args.add_proportional_weights:
        print(add_proportional_weights(args.out, args.weight))
        return
    base, out = ROOT / args.base, ROOT / args.out
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))

    # ---- candidate articles + the extra filters ----
    articles = [json.loads(l) for l in open(ROOT / args.candidate, encoding="utf-8")]
    ids = {a["page_id"] for a in articles}

    def wanted_lines():
        for path in sorted(WORK.glob("*.ns0.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if int(line[7:line.index(",")]) in ids:  # line starts with {"id": N,
                        yield line
    with Pool(4) as pool:
        shares = dict(pool.imap_unordered(template_share, wanted_lines(), chunksize=100))
    skeletons = Counter(opening_skeleton(a) for a in articles)
    filtered, dropped = [], Counter()
    for a in articles:
        if a["chars"] < 800:
            dropped["shorter than 800 characters"] += 1
        elif a["chars"] > 50_000:
            dropped["longer than 50,000 characters (one article would weigh too much)"] += 1
        elif shares.get(a["page_id"], 1.0) > 0.6:
            dropped["templates + tables > 60% of the wikitext"] += 1
        elif skeletons[opening_skeleton(a)] >= 20:
            dropped["mass-produced opening sentence"] += 1
        else:
            filtered.append(a)

    # ---- balanced sample ----
    by_topic = defaultdict(list)
    for a in sorted(filtered, key=lambda a: a["page_id"]):
        by_topic[a["topic"]].append(a)
    available = {t: sum(a["chars"] for a in v) for t, v in by_topic.items()}
    budgets = balanced_budgets(available, args.chars, args.cap_share)
    rng = random.Random(args.seed)
    sample = []
    for topic in sorted(by_topic):
        pool_ = by_topic[topic][:]
        rng.shuffle(pool_)
        total = 0
        for a in pool_:
            if total >= budgets[topic]:
                break
            sample.append(a)
            total += a["chars"]

    # ---- split by article, then drop held-out lines that share 50-character runs with train ----
    assignment = assign_splits([(a["page_id"], a["chars"]) for a in sample], 0.05, 0.05, f"{args.seed}:{NAME}")
    split_articles = {s: [a for a in sample if assignment[a["page_id"]] == s] for s in SPLITS}
    texts = {s: "\n\n".join(a["text"] for a in split_articles[s]) for s in SPLITS}
    train_hashes = window_hashes(texts["train"])
    train_lines = {" ".join(l.split()) for l in texts["train"].split("\n") if l.strip()}
    decon = {}
    for s in ("val", "test"):
        before = len(texts[s])
        texts[s], lines_dropped = decontaminate(texts[s], train_lines, train_hashes)
        decon[s] = {"chars_before": before, "chars_after": len(texts[s]), "lines_dropped": lines_dropped}

    # ---- dataset: exact copy of v2 + the Wikipedia source ----
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("tokenizers"))
    base_files = {p.relative_to(base): p.read_bytes() for p in base.glob("*/*.txt")}
    for s in SPLITS:
        (out / s / f"{NAME}.txt").write_text(texts[s], encoding="utf-8")
    changed = [str(p) for p, data in base_files.items() if (out / p).read_bytes() != data]
    if changed:
        raise RuntimeError(f"v2 files changed: {changed}")

    # ---- leakage checks ----
    v2_held = "\n".join((base / s / p.name).read_text(encoding="utf-8") for s in ("val", "test") for p in sorted((base / s).glob("*.txt")))
    v2_train = "\n".join((base / "train" / p.name).read_text(encoding="utf-8") for p in sorted((base / "train").glob("*.txt")))
    para = lambda t: {norm_para(p) for p in t.split("\n") if len(p.split()) >= 8}
    wiki_held = texts["val"] + "\n" + texts["test"]
    leakage = {
        "wiki_held_out_paragraphs_in_wiki_train": len(para(wiki_held) & para(texts["train"])),
        "wiki_held_out_50char_windows_in_wiki_train": round(shared_window_rate(wiki_held, train_hashes), 5),
        "v2_held_out_50char_windows_in_wiki_train": round(shared_window_rate(v2_held, train_hashes), 5),
        "wiki_held_out_50char_windows_in_v2_train": round(shared_window_rate(wiki_held, window_hashes(v2_train)), 5),
        "decontamination": decon,
    }
    print("leakage:", leakage, flush=True)

    # ---- weights ----
    v2_w = meta["sampling"]["weights_language_balanced"]
    general = sum(v2_w[n] for n in GENERAL_AR)
    scale = (general - args.weight) / general
    weights = {n: (w * scale if n in GENERAL_AR else w) for n, w in v2_w.items()} | {NAME: args.weight}

    # ---- meta + provenance ----
    vocab_size = len(json.loads((out / "vocab.json").read_text(encoding="utf-8"))["chars"])
    train_counts = Counter("\n".join((out / "train" / p.name).read_text(encoding="utf-8") for p in (out / "train").glob("*.txt")))
    for s in ("val", "test"):
        meta["eval_sets"][f"A_wikipedia_{s}"] = {"chars": len(texts[s]),
                                                  "unigram_baseline_loss": round(unigram_loss(train_counts, texts[s], vocab_size), 4)}
    topic_chars = Counter()
    for a in sample:
        topic_chars[a["topic"]] += a["chars"]
    meta["sources"][NAME] = {
        "language": "ar", "eval_group": "wikipedia", "private": False,
        "license": "CC-BY-SA-4.0 (Arabic Wikipedia; see data/processed/wikipedia_ar_candidate/ATTRIBUTION.md)",
        "dump_date": "2026-08-01", "articles": len(sample),
        "articles_per_split": {s: len(v) for s, v in split_articles.items()},
        **{f"{s}_chars": len(texts[s]) for s in SPLITS},
        "arabic_letters": len(ARABIC.findall("".join(texts.values()))),
        "topic_chars": dict(topic_chars.most_common()), "topic_budgets": {t: round(b) for t, b in budgets.items()},
        "extra_filters_dropped": dict(dropped), "provenance": f"{NAME}_provenance.jsonl",
    }
    meta["sampling"]["weights_language_balanced_v2"] = v2_w
    meta["sampling"]["weights_language_balanced"] = {n: round(w, 6) for n, w in weights.items()}
    meta["sampling"]["wikipedia_note"] = (f"Wikipedia {args.weight:.2%} of all batches, taken from the general Arabic "
                                          f"sources only (scaled by {scale:.4f}); books, English and After-Sales unchanged")
    meta["leakage_checks_wikipedia"] = leakage
    meta["derived_from"] = {"dataset": args.base, "fingerprint": meta["dataset_fingerprint"], "unchanged_files": len(base_files)}
    meta["totals"]["train_chars"] = sum(len((out / "train" / p.name).read_text(encoding="utf-8")) for p in (out / "train").glob("*.txt"))
    with open(out / f"{NAME}_provenance.jsonl", "w", encoding="utf-8") as f:
        for a in sample:
            f.write(json.dumps({k: a[k] for k in ("page_id", "revision_id", "revision_timestamp", "title", "url",
                                                  "source_chunk", "file", "dump_date", "topic", "chars")}
                               | {"split": assignment[a["page_id"]]}, ensure_ascii=False) + "\n")
    shutil.copy(ROOT / "data/processed/wikipedia_ar_candidate/ATTRIBUTION.md", out / "WIKIPEDIA_ATTRIBUTION.md")
    meta["dataset_fingerprint"] = fingerprint(out)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta["sources"][NAME], ensure_ascii=False, indent=1))
    print("weights:", meta["sampling"]["weights_language_balanced"])
    print("Arabic total:", round(sum(w for n, w in weights.items() if n in GENERAL_AR + ("hindawi_books_ar", NAME)), 4))
    print("fingerprint", meta["dataset_fingerprint"])


if __name__ == "__main__":
    main()
