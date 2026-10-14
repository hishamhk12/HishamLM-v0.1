"""Pass 2: clean, measure, deduplicate and sample Arabic Wikipedia articles (candidate only, no training).

Usage (from the project root), after data.wikipedia_extract:
    python -m data.wikipedia_candidate

Reads data/processed/wikipedia_ar_work/*.jsonl and writes data/processed/wikipedia_ar_candidate/:
    articles.jsonl      every eligible article after cleaning and dedup, with provenance (CC BY-SA 4.0)
    option_A.txt / option_B.txt   page ids of the ~20M / ~50M Arabic-character samples
    stats.json          all numbers of data/wikipedia_ar/REPORT.md
    ATTRIBUTION.md      license requirements for any dataset built from it
Nothing here touches the official datasets.
"""

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data.wikipedia_clean import mediawiki_to_text

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/processed/wikipedia_ar_work"
OUT = ROOT / "data/processed/wikipedia_ar_candidate"
DUMP_DATE = "2026-08-01"

ARABIC = re.compile(r"[ء-يٱ-ۓ]")
LATIN = re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[^\W\d_]")
WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|[ً-ْٰـ])*")
MARKS = re.compile(r"[ً-ْٰـ]")
INVISIBLE = re.compile(r"[​‍-‏‪-‮⁦-⁩﻿­]")
ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
EMPTY_PARENS = re.compile(r"\(\s*(?:[^()\s]{1,20}:)?\s*(?:أو|و|،)?\s*\)")
YEAR_TITLE = re.compile(r"^(\d{1,4}|[٠-٩]{1,4})( (ق\.م|ق م|م|هـ))?$|^عقد \d+|^القرن \S+$")
DATE_TITLE = re.compile(r"^\d{1,2} (يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)$")
DISAMBIG_TEMPLATES = {"توضيح", "صفحة توضيح", "disambiguation", "disambig", "توضيح جغرافي", "توضيح أسماء"}

# topic from the article's own categories (keyword in a category name -> topic); first match wins
TOPICS = [
    ("سيرة", ("مواليد", "وفيات", "أشخاص", "لاعبو", "لاعبات", "ممثلو", "ممثلات", "سياسيون", "كتاب ", "شعراء", "مغنو")),
    ("طب", ("طب", "أمراض", "أدوية", "تشريح", "صحة", "علاج", "أعراض", "جراحة")),
    ("تقنية", ("حوسبة", "برمجيات", "حاسوب", "إنترنت", "تقنية", "تكنولوجيا", "هواتف", "ألعاب فيديو", "إلكترونيات", "شبكات")),
    ("اقتصاد", ("اقتصاد", "شركات", "بنوك", "تجارة", "صناعة", "مالية", "أسواق", "عملات")),
    ("أدب", ("أدب", "روايات", "قصص", "شعر", "كتب ", "مسرحيات")),
    ("علم", ("علم", "علوم", "فيزياء", "كيمياء", "رياضيات", "فلك", "أحياء", "حيوانات", "نباتات", "أنواع", "حشرات",
             "طيور", "أسماك", "معادن", "جيولوجيا", "كويكبات", "نجوم", "مجرات")),
    ("تاريخ", ("تاريخ", "حروب", "معارك", "سلالات", "إمبراطوريات", "عقد ", "القرن", "ممالك", "ثورات")),
    ("جغرافيا", ("مدن", "قرى", "بلديات", "مقاطعات", "بلدات", "جغرافيا", "أنهار", "جبال", "جزر", "بحيرات", "تجمعات سكانية",
                 "مناطق", "محافظات", "أحياء ")),
    ("ثقافة", ("أفلام", "مسلسلات", "موسيقى", "ألبومات", "أغاني", "فن", "رياضة", "كرة", "أندية", "دين", "مساجد",
               "كنائس", "ثقافة", "تلفزيون", "مهرجانات", "بطولات", "فرق")),
]


def topic_of(categories):
    for cat in categories:
        for topic, keys in TOPICS:
            if any(k in cat for k in keys):
                return topic
    return "عام / غير مصنف"


def norm_para(p):
    return " ".join(MARKS.sub("", p).split())


def letters_share(text):
    ar, la, all_ = len(ARABIC.findall(text)), len(LATIN.findall(text)), len(LETTER.findall(text))
    return ar / max(all_, 1), la / max(all_, 1), (all_ - ar - la) / max(all_, 1)


# ---------------------------------------------------------------------------
# Cleaning one article (run in a process pool)
# ---------------------------------------------------------------------------

def process(line):
    row = json.loads(line)
    stats, names = Counter(), Counter()
    text, categories = mediawiki_to_text(row["wikitext"], stats, names)
    raw = len(row["wikitext"])
    ar, la, other = letters_share(text)
    paras = [p for p in text.split("\n") if p.strip()]
    norm = Counter(norm_para(p) for p in paras if len(p.split()) >= 8)
    title = row["title"]
    return {
        "id": row["id"], "rev_id": row["rev_id"], "timestamp": row["timestamp"], "title": title, "chunk": row["chunk"],
        "revisions": row["revisions"], "raw_chars": raw, "chars": len(text), "words": len(WORD.findall(text)),
        "arabic_share": ar, "latin_share": la, "other_share": other, "categories": categories[:30],
        "topic": topic_of(categories),
        "is_list": title.startswith(("قائمة ", "قوائم ")),
        "is_disambiguation": title.endswith("(توضيح)") or any(n.lower() in DISAMBIG_TEMPLATES for n in names),
        "is_year_or_date": bool(YEAR_TITLE.match(title) or DATE_TITLE.match(title)),
        "template_table_share": (stats["template_chars"] + stats["table_chars"]) / max(raw, 1),
        "templates": stats["template_count"], "tables": stats["table_count"],
        "broken_markup_lines": stats["broken_markup_lines"], "unbalanced": stats["unbalanced_template"] + stats["unbalanced_table"],
        "entities_left": len(ENTITY.findall(text)), "presentation_forms": stats["presentation_forms"],
        "invisible_chars": len(INVISIBLE.findall(text)), "empty_parens": len(EMPTY_PARENS.findall(text)),
        "repeated_paragraphs": sum(c - 1 for c in norm.values() if c > 1),
        "paragraphs": len(paras),
        "paragraphs_without_end_punct": sum(not p.rstrip().endswith((".", "؟", "!", ":", "»", ")", "؛")) and len(p.split()) >= 8 for p in paras),
        "text": text,
    }


# ---------------------------------------------------------------------------
# Near duplicates (MinHash LSH on word 5-grams, verified with exact Jaccard)
# ---------------------------------------------------------------------------

def near_duplicate_pairs(texts, n=5, perms=64, bands=16, threshold=0.8, seed=0):
    rng = np.random.default_rng(seed)
    prime = np.uint64((1 << 61) - 1)
    a = rng.integers(1, (1 << 61) - 1, perms, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, perms, dtype=np.uint64)
    rows = perms // bands
    shingle_sets, buckets = [], defaultdict(list)
    for idx, t in enumerate(texts):
        w = norm_para(t).split()
        s = {hash(" ".join(w[i:i + n])) & ((1 << 61) - 1) for i in range(max(1, len(w) - n + 1))}
        shingle_sets.append(s)
        if len(s) < 10:
            continue
        h = np.fromiter(s, dtype=np.uint64)
        sig = ((np.outer(h, a) + b) % prime).min(axis=0)
        for band in range(bands):
            buckets[(band, sig[band * rows:(band + 1) * rows].tobytes())].append(idx)
    pairs = set()
    for ids in buckets.values():
        if 1 < len(ids) <= 200:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.add((ids[i], ids[j]))
    result = []
    for i, j in pairs:
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if len(s1 & s2) / len(s1 | s2) >= threshold:
            result.append((i, j))
    return result


# ---------------------------------------------------------------------------
# Overlap with our data
# ---------------------------------------------------------------------------

_HAYSTACK = None


def _init(haystack):
    global _HAYSTACK
    _HAYSTACK = haystack


def _found(window):
    return window in _HAYSTACK


def windows_found(windows, haystack, workers=4):
    with Pool(workers, initializer=_init, initargs=(haystack,)) as pool:
        return sum(pool.map(_found, windows, chunksize=50))


def read_split_files(dataset, names, splits=("train", "val", "test")):
    return "\n".join((ROOT / dataset / s / f"{n}.txt").read_text(encoding="utf-8")
                     for s in splits for n in names if (ROOT / dataset / s / f"{n}.txt").exists())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"dump_date": DUMP_DATE, "chunks": {}}

    # ---- pages (all namespaces) ----
    for path in sorted(WORK.glob("*.pages.jsonl")):
        chunk = path.name.removesuffix(".pages.jsonl")
        ns, ns0_redirects, revisions, latest = Counter(), 0, 0, ""
        for line in open(path, encoding="utf-8"):
            p = json.loads(line)
            ns[p["ns"]] += 1
            revisions += p["revisions"]
            ns0_redirects += p["ns"] == 0 and p["redirect"]
            latest = max(latest, p["timestamp"] or "")
        stats["chunks"][chunk] = {"pages": sum(ns.values()), "namespaces": dict(ns.most_common()),
                                  "ns0_pages": ns[0], "ns0_redirects": ns0_redirects,
                                  "ns0_articles": ns[0] - ns0_redirects, "revisions": revisions,
                                  "latest_revision": latest}
        print(chunk, stats["chunks"][chunk], flush=True)

    # ---- clean every article ----
    def lines():
        for path in sorted(WORK.glob("*.ns0.jsonl")):
            with open(path, encoding="utf-8") as f:
                yield from f
    with Pool(4) as pool:
        articles = list(pool.imap(process, lines(), chunksize=200))
    print(f"cleaned {len(articles):,} articles", flush=True)

    # ---- duplicates inside Wikipedia ----
    para_first = {}
    exact_dup_paras = exact_dup_chars = 0
    for i, a in enumerate(articles):
        for p in a["text"].split("\n"):
            if len(p.split()) >= 8:
                key = norm_para(p)
                if key in para_first and para_first[key] != i:
                    exact_dup_paras += 1
                    exact_dup_chars += len(p)
                else:
                    para_first.setdefault(key, i)
    long_enough = [i for i, a in enumerate(articles) if a["words"] >= 50]
    near = near_duplicate_pairs([articles[i]["text"] for i in long_enough])
    near_dup = {max(long_enough[i], long_enough[j]) for i, j in near}  # keep the first of each pair
    # mass-produced intros: first paragraph with the title and the numbers masked
    def skeleton(a):
        first = next((p for p in a["text"].split("\n") if len(p.split()) >= 5), "")
        s = first.replace(a["title"].split(" (")[0], "T")
        return re.sub(r"[0-9٠-٩]+", "0", norm_para(s))
    skeletons = Counter(skeleton(a) for a in articles)
    for a in articles:
        a["templated_intro"] = skeletons[skeleton(a)] >= 20

    # ---- eligibility ----
    def reason(i, a):
        if a["is_disambiguation"]:
            return "disambiguation"
        if a["is_list"]:
            return "list"
        if a["is_year_or_date"]:
            return "year / date"
        if a["chars"] < 400:
            return "very short (< 400 clean chars)"
        if a["arabic_share"] < 0.8:
            return "less than 80% Arabic letters"
        if i in near_dup:
            return "near duplicate of another article"
        return None
    excluded = Counter()
    for i, a in enumerate(articles):
        r = reason(i, a)
        a["exclusion"] = r
        if r:
            excluded[r] += 1

    # ---- overlap with our data ----
    refs = {
        "dataset v2 (all files)": "\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "data/processed/hishamlm_v2").glob("*/*.txt"))),
        "Hindawi books": read_split_files("data/processed/hishamlm_v2", ["hindawi_books_ar"]),
        "new books (Kalila, Nazarat, Bayan)": read_split_files("data/processed/hishamlm_v3", ["kalila_wa_dimna", "nazarat_al_bakhil", "al_bayan_wal_tabyin"]),
        "general Arabic (CV, NTREX, XQuAD)": read_split_files("data/processed/hishamlm_v2", ["common_voice_ar", "ntrex_ar", "xquad_ar"]),
        "After-Sales": read_split_files("data/processed/hishamlm_v2", ["after_sales"]),
    }
    eligible = [a for a in articles if not a["exclusion"]]
    ref_paras = {k: {norm_para(l) for l in v.split("\n") if len(l.split()) >= 8} for k, v in refs.items()}
    overlap = {}
    rng = random.Random(0)
    sample_windows = []
    for a in rng.sample(eligible, min(4000, len(eligible))):
        t = a["text"]
        for _ in range(5):
            s = rng.randrange(max(1, len(t) - 50))
            sample_windows.append(t[s:s + 50])
    wiki_paras = {norm_para(p) for a in eligible for p in a["text"].split("\n") if len(p.split()) >= 8}
    for name, ref in refs.items():
        overlap[name] = {
            "exact_paragraphs_shared": len(wiki_paras & ref_paras[name]),
            "wiki_50char_windows_found_in_ref": round(sum(w in ref for w in sample_windows) / len(sample_windows), 5),
        }
    # the other direction: our held-out text found in Wikipedia (would be leakage if Wikipedia were trained on)
    haystack = "\n".join(a["text"] for a in eligible)
    held_out = {
        "A general Arabic val+test": read_split_files("data/processed/hishamlm_v2", ["common_voice_ar", "ntrex_ar", "xquad_ar"], ("val", "test")),
        "Hindawi books val+test": read_split_files("data/processed/hishamlm_v2", ["hindawi_books_ar"], ("val", "test")),
        "new books val+test": read_split_files("data/processed/hishamlm_v3", ["kalila_wa_dimna", "al_bayan_wal_tabyin"], ("val", "test")),
        "After-Sales val+test": read_split_files("data/processed/hishamlm_v2", ["after_sales"], ("val", "test")),
    }
    for name, text in held_out.items():
        windows = [text[i:i + 50] for i in range(0, len(text) - 50, 50)]
        windows = rng.sample(windows, min(1000, len(windows)))
        overlap.setdefault("held_out_in_wikipedia", {})[name] = round(windows_found(windows, haystack) / len(windows), 5)
    wiki_para_set_by_ref = {k: ref_paras[k] for k in refs}
    affected = [a for a in eligible if any(norm_para(p) in wiki_para_set_by_ref["dataset v2 (all files)"]
                                           for p in a["text"].split("\n") if len(p.split()) >= 8)]
    overlap["articles_sharing_a_paragraph_with_v2"] = len(affected)
    del haystack

    # ---- final candidate text: drop repeated paragraphs and paragraphs found in our data ----
    seen, ours = set(), set().union(*ref_paras.values())
    dropped_paras = Counter()
    for a in eligible:
        kept = []
        for p in a["text"].split("\n"):
            key = norm_para(p) if len(p.split()) >= 8 else None
            if key and key in ours:
                dropped_paras["also in our datasets"] += 1
                continue
            if key and key in seen:
                dropped_paras["repeated in Wikipedia"] += 1
                continue
            if key:
                seen.add(key)
            kept.append(p)
        a["text"] = "\n".join(kept).strip()
        a["chars"] = len(a["text"])
        a["arabic_chars"] = len(ARABIC.findall(a["text"]))
    eligible = [a for a in eligible if a["chars"] >= 400]

    # ---- BPE (old 4K) on a sample ----
    from tokenizer import load_tokenizer
    tok = load_tokenizer(ROOT / "data/processed/hishamlm_v1/tokenizers/bpe_4096.json")
    sample_text = "\n\n".join(a["text"] for a in rng.sample(eligible, min(3000, len(eligible))))[:6_000_000]
    ids = tok.encode(sample_text)
    char_len = tok.token_char_lengths()
    byte_ids = [i for i in ids if i < 256]
    rare = Counter(ch for ch in sample_text if ch not in tok.char_to_id and not ch.isspace())
    bpe = {"sample_chars": len(sample_text), "tokens": len(ids), "chars_per_token": round(len(sample_text) / len(ids), 3),
           "byte_fallback_token_rate": round(len(byte_ids) / len(ids), 5),
           "byte_fallback_char_rate": round(sum(char_len[i] for i in byte_ids) / len(sample_text), 5),
           "rare_characters_top": rare.most_common(15), "distinct_rare_characters": len(rare),
           "vocab_used": round(len(set(ids)) / tok.vocab_size, 4),
           "merged_token_share": round(sum(char_len[i] >= 2 for i in ids) / len(ids), 4)}

    # ---- options A / B: random articles until the Arabic-character budget ----
    order = sorted(eligible, key=lambda a: a["id"])
    random.Random(42).shuffle(order)
    options = {}
    for name, budget in (("A", 20_000_000), ("B", 50_000_000)):
        chosen, total = [], 0
        for a in order:
            if total >= budget:
                break
            chosen.append(a)
            total += a["arabic_chars"]
        chars = [a["chars"] for a in chosen]
        options[name] = {
            "arabic_chars_target": budget, "articles": len(chosen), "chars": sum(chars), "arabic_chars": total,
            "pool_exhausted": total < budget,
            "mean_chars": round(statistics.mean(chars)), "median_chars": statistics.median(chars),
            "arabic_share_of_letters": round(sum(a["arabic_share"] * a["chars"] for a in chosen) / sum(chars), 4),
            "topics": dict(Counter(a["topic"] for a in chosen).most_common()),
            "topic_chars": {t: sum(a["chars"] for a in chosen if a["topic"] == t) for t, _ in Counter(a["topic"] for a in chosen).most_common()},
            "templated_intro_share": round(sum(a["templated_intro"] for a in chosen) / len(chosen), 4),
            "estimated_bpe_tokens": round(sum(chars) / bpe["chars_per_token"]),
        }
        (OUT / f"option_{name}.txt").write_text("\n".join(str(a["id"]) for a in chosen) + "\n", encoding="utf-8")

    # ---- write the candidate ----
    with open(OUT / "articles.jsonl", "w", encoding="utf-8") as f:
        for a in eligible:
            f.write(json.dumps({
                "source": "Arabic Wikipedia (ar.wikipedia.org)", "license": "CC BY-SA 4.0", "dump_date": DUMP_DATE,
                "file": a["chunk"] + ".xml.bz2", "source_chunk": a["chunk"], "page_id": a["id"], "revision_id": a["rev_id"],
                "revision_timestamp": a["timestamp"], "title": a["title"],
                "url": "https://ar.wikipedia.org/wiki/" + a["title"].replace(" ", "_"),
                "history_url": f"https://ar.wikipedia.org/w/index.php?curid={a['id']}&action=history",
                "topic": a["topic"], "chars": a["chars"], "text": a["text"]}, ensure_ascii=False) + "\n")

    # ---- statistics ----
    arts = articles
    lens = [a["chars"] for a in arts]
    buckets = Counter()
    for n in lens:
        buckets[next(lbl for lim, lbl in ((200, "<200"), (500, "200-500"), (1000, "500-1K"), (2000, "1K-2K"),
                                          (5000, "2K-5K"), (10000, "5K-10K"), (50000, "10K-50K"), (10**12, "50K+")) if n < lim)] += 1
    total_letters = sum(len(LETTER.findall(a["text"])) for a in arts) or 1
    by_size = sorted(arts, key=lambda a: a["chars"])
    stats.update({
        "articles_cleaned": len(arts),
        "raw_wikitext_chars": sum(a["raw_chars"] for a in arts),
        "clean_chars": sum(lens), "clean_words": sum(a["words"] for a in arts),
        "letters": {"arabic": round(sum(a["arabic_share"] * a["chars"] for a in arts) / sum(lens), 4),
                    "latin": round(sum(a["latin_share"] * a["chars"] for a in arts) / sum(lens), 4),
                    "other": round(sum(a["other_share"] * a["chars"] for a in arts) / sum(lens), 4)},
        "length": {"mean": round(statistics.mean(lens)), "median": statistics.median(lens),
                   "histogram": {k: buckets[k] for k in ("<200", "200-500", "500-1K", "1K-2K", "2K-5K", "5K-10K", "10K-50K", "50K+")}},
        "largest": [(a["title"], a["chars"]) for a in by_size[-5:][::-1]],
        "smallest": [(a["title"], a["chars"]) for a in by_size[:5]],
        "very_short_under_400": sum(n < 400 for n in lens),
        "lists": sum(a["is_list"] for a in arts), "disambiguation": sum(a["is_disambiguation"] for a in arts),
        "years_dates": sum(a["is_year_or_date"] for a in arts),
        "template_or_table_heavy_over_60pct": sum(a["template_table_share"] > 0.6 for a in arts),
        "articles_with_tables": sum(a["tables"] > 0 for a in arts),
        "templated_intro_articles": sum(a["templated_intro"] for a in arts),
        "top_templated_intros": [(s[:120], c) for s, c in skeletons.most_common(8)],
        "quality": {k: sum(a[k] for a in arts) for k in ("broken_markup_lines", "unbalanced", "entities_left",
                                                          "presentation_forms", "invisible_chars", "empty_parens",
                                                          "repeated_paragraphs", "paragraphs", "paragraphs_without_end_punct")},
        "duplicates": {"exact_duplicate_paragraphs": exact_dup_paras, "exact_duplicate_paragraph_chars": exact_dup_chars,
                       "near_duplicate_pairs": len(near), "near_duplicate_articles": len(near_dup),
                       "near_duplicate_rate": round(len(near_dup) / max(len(long_enough), 1), 4)},
        "excluded": dict(excluded), "dropped_paragraphs": dict(dropped_paras),
        "candidate": {"articles": len(eligible), "chars": sum(a["chars"] for a in eligible),
                      "arabic_chars": sum(a["arabic_chars"] for a in eligible),
                      "topics": dict(Counter(a["topic"] for a in eligible).most_common())},
        "overlap": overlap, "bpe_old_4k": bpe, "options": options,
    })
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- samples for review: one short excerpt per topic ----
    samples = {}
    for a in rng.sample(eligible, len(eligible)):
        if a["topic"] not in samples and 1500 <= a["chars"]:
            samples[a["topic"]] = {"title": a["title"], "page_id": a["id"], "excerpt": a["text"][:350]}
    (OUT / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("articles_cleaned", "clean_chars", "letters", "length", "duplicates",
                                            "excluded", "candidate", "overlap", "bpe_old_4k", "options")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
