"""Generated-text measures for the Wikipedia 20M comparison (baseline v2 vs v2 + Wikipedia).

Usage (from the project root):
    python -m inference.wiki_style_eval --samples base=checkpoints/A/samples_best_long_ar.json \\
        wiki=checkpoints/B/samples_best_long_ar.json "base_coh=coherence.json#baseline" ... --out wiki_style.json

For every set of generations (and for real held-out text as reference):
  invented words      share of Arabic words not in the training lexicon (dataset v2 + Wikipedia train, the same
                      lexicon for both models), and the same after accepting words from Arabic Wikipedia titles
                      (data/wikipedia_ar/raw/arwiki-latest-all-titles-in-ns0.gz; evaluation only, never trained on)
  style / drift       per 100 Arabic words: Hindawi-books vocabulary, new-books (Kalila / al-Bayan) vocabulary,
                      Wikipedia vocabulary (words over-represented in Wikipedia train against the current Arabic),
                      and the hand-picked Wikipedia markers below; encyclopedic patterns per 1000 characters
  structure           words per sentence, fragments (< 4 words), repeated sentence openings
  memorization        longest verbatim match with all training text, 50-character windows found in it,
                      generated paragraphs found in it, openings copied from Wikipedia articles; the source
                      (and the Wikipedia article) of every match longer than 40 characters
"""

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from inference.coherence import normalize, sentences, stem, words
from inference.memorization import longest_match, ngram_overlap
from inference.vocab_drift import GENERAL, counts, distinctive

ROOT = Path(__file__).resolve().parents[1]
ARABIC = re.compile("[ء-ي]")
MARKERS = ["هو", "هي", "يعد", "تعد", "ولد", "توفي", "تقع", "يعتبر", "يبلغ", "حسب", "نسبة", "سكان", "مقاطعة", "مدينة"]
PATTERNS = {
    "(مواليد …)": r"\(\s*مواليد",
    "X هو/هي أحد/مدينة/لاعب…": r"\b(?:هو|هي) (?:أحد|إحدى|مدينة|قرية|بلدة|لاعب|لاعبة|ممثل|ممثلة|كاتب|كاتبة|سياسي|مغني|"
                               r"مغنية|عالم|نوع|جنس|فيلم|رواية|مسلسل|شركة|منطقة|مقاطعة|نهر|جبل)\b",
    "تقع في": r"\bتقع في\b",
    "يبلغ عدد سكان": r"\bيبلغ عدد سكان",
    "وُلد في / توفي في": r"\b(?:ولد|توفي|توفيت|ولدت) (?:في|عام)\b",
}


def arabic_words(text):
    return [w for w in words(normalize(text)) if ARABIC.match(w)]


def marker_rate(ws):
    hits = sum(w in MARKERS or (w[:1] in "وف" and w[1:] in MARKERS) for w in ws)
    return 100 * hits / max(len(ws), 1)


def measures(texts, lexicon, names, lists, train_text, openings, attribution):
    rows = []
    for t in texts:
        ws = arabic_words(t)
        stems = [stem(w) for w in ws if len(w) >= 3]
        sents = sentences(t)
        heads = [tuple(arabic_words(s)[:2]) for s in sents]
        unknown = [w for w in ws if w not in lexicon]
        row = {
            "invented_old": len(unknown) / max(len(ws), 1),
            "invented_new": sum(w not in names for w in unknown) / max(len(ws), 1),
            "words_per_sentence": len(ws) / max(len(sents), 1),
            "fragments": sum(len(arabic_words(s)) < 4 for s in sents) / max(len(sents), 1),
            "repeated_openings": sum(heads[i] in heads[:i] for i in range(len(heads)) if len(heads[i]) == 2) / max(len(sents), 1),
            "wiki_markers_per_100w": marker_rate(ws),
            **{f"{k}_per_100w": 100 * sum(s in v for s in stems) / max(len(stems), 1) for k, v in lists.items()},
            "encyclopedic_patterns_per_1000c": 1000 * sum(len(re.findall(p, normalize(t))) for p in PATTERNS.values()) / max(len(t), 1),
        }
        if train_text is not None:
            length, match = longest_match(t, train_text)
            paras = [p for p in t.split("\n") if len(p.split()) >= 8]
            row |= {"longest_match": length, "overlap_50char": ngram_overlap(t, train_text, 50, 5),
                    "paragraphs_in_train": sum(" ".join(p.split()) in train_text for p in paras),
                    "copied_wikipedia_openings": sum(" ".join(arabic_words(s)[:5]) in openings for s in sents
                                                     if len(arabic_words(s)) >= 5)}
            if length > 40:
                row["match"] = {"chars": length, "text": match[:80], "source": attribution(match)}
        rows.append(row)
    keys = [k for k in rows[0] if k != "match"]
    summary = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys} | {"samples": len(rows)}
    if train_text is not None:
        summary["longest_match_max"] = max(r["longest_match"] for r in rows)
        summary["matches_over_40"] = [r["match"] for r in rows if "match" in r]
    return summary


def load(spec):
    label, path = spec.split("=", 1)
    path, _, model = path.partition("#")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if model:
        model, _, groups = model.partition("@")
        groups = set(groups.split(",")) if groups else {"arabic", "books"}
        return label, [s["continuation"] for s in data["models"][model]["samples"] if s["group"] in groups]
    return label, [r["continuation"] for r in data["results"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True, help="label=samples.json or label=coherence.json#model[@groups]")
    parser.add_argument("--dataset", default="data/processed/hishamlm_v2_wiki20m")
    parser.add_argument("--books-dataset", default="data/processed/hishamlm_v3")
    parser.add_argument("--titles", default="data/wikipedia_ar/raw/arwiki-latest-all-titles-in-ns0.gz")
    parser.add_argument("--candidate", default="data/processed/wikipedia_ar_candidate/articles.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ds, books = ROOT / args.dataset, ROOT / args.books_dataset
    train_files = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ds / "train").glob("*.txt"))}
    train_text = "\n".join(train_files.values())
    lexicon = set(arabic_words(train_text))
    with gzip.open(ROOT / args.titles, "rt", encoding="utf-8") as f:
        names = {w for line in f for w in arabic_words(line.replace("_", " "))} - lexicon

    current_ar = counts(ds, GENERAL) + counts(ds, ["hindawi_books_ar"])
    lists = {
        "hindawi": set(distinctive(counts(ds, ["hindawi_books_ar"]), counts(ds, GENERAL), 30, 8.0, 300)),
        "new_books": set(distinctive(counts(books, ["kalila_wa_dimna", "al_bayan_wal_tabyin", "nazarat_al_bakhil"]),
                                     current_ar, 30, 8.0, 300)),
        "wikipedia": set(distinctive(counts(ds, ["wikipedia_ar"]), current_ar, 30, 8.0, 300)),
    }

    # Wikipedia train articles: openings (first 5 words of every sentence) and attribution of long matches
    provenance = {r["page_id"]: r["split"] for r in map(json.loads, open(ds / "wikipedia_ar_provenance.jsonl", encoding="utf-8"))}
    wiki_train = {}
    for line in open(ROOT / args.candidate, encoding="utf-8"):
        a = json.loads(line)
        if provenance.get(a["page_id"]) == "train":
            wiki_train[a["title"]] = a["text"]
    openings = {" ".join(arabic_words(s)[:5]) for t in wiki_train.values() for s in sentences(t) if len(arabic_words(s)) >= 5}

    def attribution(match):
        for name, text in train_files.items():
            if match in text:
                if name == "wikipedia_ar":
                    title = next((t for t, body in wiki_train.items() if match in body), None)
                    return {"source": name, "article": title}
                return {"source": name}
        return None

    report = {"settings": vars(args), "lexicon_words": len(lexicon), "names_from_titles": len(names),
              "lists": {k: sorted(v) for k, v in lists.items()}, "markers": MARKERS, "patterns": PATTERNS, "results": {}}
    for spec in args.samples:
        label, texts = load(spec)
        # coherence.json already holds memorization (inference.coherence, same training text): not repeated here
        memo_text = None if "#" in spec else train_text
        report["results"][label] = measures(texts, lexicon, names, lists, memo_text, openings, attribution)
        print(label, {k: v for k, v in report["results"][label].items() if k != "matches_over_40"}, flush=True)

    # references: real held-out text cut to 400-character pieces
    refs = {"real general Arabic (test)": "\n".join((ds / "test" / f"{n}.txt").read_text(encoding="utf-8") for n in GENERAL),
            "real Hindawi books (test)": (ds / "test" / "hindawi_books_ar.txt").read_text(encoding="utf-8"),
            "real Wikipedia (test)": (ds / "test" / "wikipedia_ar.txt").read_text(encoding="utf-8")}
    for label, text in refs.items():
        pieces = [text[i:i + 400] for i in range(0, len(text) - 400, max(400, len(text) // 150))][:150]
        report["results"][label] = measures(pieces, lexicon, names, lists, None, openings, attribution)
        print(label, report["results"][label], flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
