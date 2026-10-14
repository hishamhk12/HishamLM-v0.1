"""Audit of the Arabic Wikipedia title files (no article text, no training).

Usage (from the project root):
    python -m data.wikipedia_titles_audit

Reads, streaming:
  data/wikipedia_ar/raw/arwiki-latest-all-titles-in-ns0.gz
      one title per line (spaces written as "_"), namespace 0 only; redirects are included and not marked
  data/wikipedia_ar/raw/arwiki-latest-pages-articles-multistream-index.txt.bz2
      "offset:page_id:title" for every page of pages-articles-multistream.xml.bz2 (all content namespaces);
      offset = byte position of the bz2 stream (about 100 pages each) that holds the page
Writes data/wikipedia_ar/audit.json (counts and short examples only).
"""

import bz2
import gzip
import html
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/wikipedia_ar/raw"
TITLES = RAW / "arwiki-latest-all-titles-in-ns0.gz"
INDEX = RAW / "arwiki-latest-pages-articles-multistream-index.txt.bz2"

# namespaces seen in the index (prefix before ":"); everything else is namespace 0 (articles)
NAMESPACES = {"تصنيف": "Category", "قالب": "Template", "ملف": "File", "ويكيبيديا": "Wikipedia (project)",
              "بوابة": "Portal", "موضوع": "Topic (talk threads)", "وحدة": "Module (Lua)",
              "ميدياويكي": "MediaWiki (interface)", "مساعدة": "Help", "فعالية": "Event"}

ARABIC = re.compile(r"[ء-يٱ-ۓ]")
LATIN = re.compile(r"[A-Za-z]")
OTHER_LETTER = re.compile(r"[^\W\d_]")
INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
MARKS = re.compile(r"[ً-ْٰـ]")
YEARISH = re.compile(r"^(\d{1,4}|[٠-٩]{1,4})( (ق\.م|ق م|م|هـ|هجري|ميلادي|في [^ ]+))?$")
DATE = re.compile(r"^\d{1,2} (يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)$")
LIST = re.compile(r"^(قائمة|قوائم) ")
DISAMBIG = re.compile(r"\(توضيح\)$")
QUALIFIER = re.compile(r"\(([^()]+)\)$")


def script(title):
    ar, la = bool(ARABIC.search(title)), bool(LATIN.search(title))
    if ar and la:
        return "mixed Arabic + Latin"
    if ar:
        return "Arabic"
    if la:
        return "Latin"
    if OTHER_LETTER.search(title):
        return "other script"
    return "no letters (digits / symbols)"


def normalize(title):
    """Spelling-insensitive key: hamza/alef forms, ta marbuta, alef maqsura, case, punctuation, spaces."""
    t = MARKS.sub("", INVISIBLE.sub("", title)).lower()
    t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"[\W_]+", "", t)


def percentiles(values):
    q = statistics.quantiles(values, n=100)
    return {"min": min(values), "p10": q[9], "p50": q[49], "p90": q[89], "p99": q[98], "max": max(values),
            "mean": round(statistics.mean(values), 2)}


def tokenizer_stats(arabic_titles, sample=100_000):
    """Old BPE 4K on a sample of Arabic titles, and how many title words the current corpus already has."""
    from tokenizer import load_tokenizer

    tok = load_tokenizer(ROOT / "data/processed/hishamlm_v1/tokenizers/bpe_4096.json")
    titles = random.Random(2).sample(sorted(arabic_titles), min(sample, len(arabic_titles)))
    text = "\n".join(titles)
    ids = tok.encode(text)
    char_len = tok.token_char_lengths()
    byte_ids = [i for i in ids if i < 256]
    v2 = ROOT / "data/processed/hishamlm_v2/train"
    lexicon = {MARKS.sub("", w) for n in ("common_voice_ar", "ntrex_ar", "xquad_ar", "hindawi_books_ar")
               for w in re.findall(r"[\u0621-\u064A\u064B-\u0652\u0670\u0640]+", (v2 / f"{n}.txt").read_text(encoding="utf-8"))}
    words = [MARKS.sub("", w) for w in re.findall(r"[\u0621-\u064A\u064B-\u0652\u0670\u0640]+", text)]
    distinct = set(words)
    return {"sample_titles": len(titles), "chars_per_token": round(len(text) / len(ids), 3),
            "byte_fallback_char_rate": round(sum(char_len[i] for i in byte_ids) / len(text), 5),
            "fallback_characters": Counter(ch for ch in text if ch not in tok.char_to_id).most_common(10),
            "title_words_known_to_v2_train": round(sum(w in lexicon for w in words) / len(words), 4),
            "distinct_title_words": len(distinct),
            "distinct_title_words_new_to_v2_train": len(distinct - lexicon)}


def main():
    # ---- all-titles ----
    with gzip.open(TITLES, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        titles = [line.rstrip("\n").replace("_", " ") for line in f]
    title_counts = Counter(titles)
    unique_titles = set(titles)

    # ---- index ----
    offsets, ids, ns_counts, index_ns0 = set(), Counter(), Counter(), []
    pages_per_offset = Counter()
    index_lines, bad = 0, 0
    with bz2.open(INDEX, "rt", encoding="utf-8") as f:
        for line in f:
            index_lines += 1
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
                bad += 1
                continue
            offset, page_id, title = int(parts[0]), int(parts[1]), html.unescape(parts[2])  # the index escapes " and &
            offsets.add(offset)
            pages_per_offset[offset] += 1
            ids[page_id] += 1
            prefix = title.split(":", 1)[0] if ":" in title else None
            ns = NAMESPACES.get(prefix, "Article (namespace 0)")
            ns_counts[ns] += 1
            if ns == "Article (namespace 0)":
                index_ns0.append(title)
    index_ns0_set = set(index_ns0)

    # ---- titles: scripts, lengths, words, qualifiers ----
    scripts = Counter(script(t) for t in unique_titles)
    examples = defaultdict(list)
    for t in random.Random(0).sample(titles, 20000):
        s = script(t)
        if len(examples[s]) < 8:
            examples[s].append(t)
    plain = [t for t in random.Random(1).sample(titles, 5000) if script(t) == "Arabic" and not QUALIFIER.search(t)][:12]
    lengths_chars = [len(t) for t in unique_titles]
    lengths_words = [len(t.split()) for t in unique_titles]
    word_buckets = Counter(min(n, 8) for n in lengths_words)
    words = Counter(w for t in unique_titles if ARABIC.search(t) for w in re.findall(r"[ء-ي]+", MARKS.sub("", t)))
    qualifiers = Counter(m.group(1) for t in unique_titles if (m := QUALIFIER.search(t)))

    # ---- duplicates and spelling variants ----
    norm_groups = defaultdict(set)
    for t in unique_titles:
        key = normalize(t)
        if key:  # titles made only of symbols have no key
            norm_groups[key].add(t)
    variant_groups = [g for g in norm_groups.values() if len(g) > 1]

    # ---- titles unsuitable as training text ----
    unsuitable = {
        "disambiguation (… (توضيح))": [t for t in unique_titles if DISAMBIG.search(t)],
        "lists (قائمة / قوائم …)": [t for t in unique_titles if LIST.match(t)],
        "years / dates only": [t for t in unique_titles if YEARISH.match(t) or DATE.match(t)],
        "no letters (digits / symbols only)": [t for t in unique_titles if script(t) == "no letters (digits / symbols)"],
        "Latin-only (in arwiki almost always a redirect from a foreign name)": [t for t in unique_titles if script(t) == "Latin"],
        "invisible / direction characters": [t for t in unique_titles if INVISIBLE.search(t)],
        "longer than 100 characters": [t for t in unique_titles if len(t) > 100],
        "contains / (subpage-like)": [t for t in unique_titles if "/" in t],
    }

    report = {
        "files": {p.name: {"bytes": p.stat().st_size} for p in (TITLES, INDEX)},
        "all_titles": {
            "header_line": header, "titles": len(titles), "unique_titles": len(unique_titles),
            "exact_duplicates": sum(c - 1 for c in title_counts.values() if c > 1),
            "duplicate_examples": [t for t, c in title_counts.most_common(5) if c > 1],
        },
        "index": {
            "lines": index_lines, "malformed_lines": bad, "unique_page_ids": len(ids),
            "page_ids_listed_twice": sum(1 for c in ids.values() if c > 1),
            "unique_offsets": len(offsets), "max_offset_bytes": max(offsets),
            "pages_per_stream": percentiles(list(pages_per_offset.values())),
            "namespaces": dict(ns_counts.most_common()),
            "namespace0_titles": len(index_ns0), "namespace0_unique_titles": len(index_ns0_set),
        },
        "comparison": {
            "in_both": len(unique_titles & index_ns0_set),
            "only_in_all_titles": len(unique_titles - index_ns0_set),
            "only_in_index_ns0": len(index_ns0_set - unique_titles),
            "examples_only_in_all_titles": sorted(unique_titles - index_ns0_set)[:10],
            "examples_only_in_index_ns0": sorted(index_ns0_set - unique_titles)[:10],
        },
        "scripts": {k: {"titles": v, "share": round(v / len(unique_titles), 4)} for k, v in scripts.most_common()},
        "script_examples": dict(examples),
        "plain_article_examples": plain,
        "length_chars": percentiles(lengths_chars),
        "length_words": percentiles(lengths_words),
        "length_words_histogram": {("8+" if k == 8 else str(k)): v for k, v in sorted(word_buckets.items())},
        "top_arabic_title_words": words.most_common(40),
        "top_parenthetical_qualifiers": qualifiers.most_common(25),
        "spelling_variant_groups": len(variant_groups),
        "titles_in_variant_groups": sum(len(g) for g in variant_groups),
        "spelling_variant_examples": [sorted(g) for g in sorted(variant_groups, key=len, reverse=True)[:3]]
                                     + [sorted(g) for g in variant_groups[:7]],
        "unsuitable_for_training": {k: {"titles": len(v), "examples": sorted(v)[:6]} for k, v in unsuitable.items()},
    }
    report["old_bpe_4k"] = tokenizer_stats([t for t in unique_titles if script(t) == "Arabic"])
    out = ROOT / "data/wikipedia_ar/audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_titles", "index", "comparison", "scripts")}, ensure_ascii=False, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
