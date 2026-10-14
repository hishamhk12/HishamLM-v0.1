"""Inspect a new batch of private Arabic EPUB books and build a candidate corpus. No training.

Usage (from the project root):
    python -m data.books_candidate --raw data/raw/private_books/books_ar_batch2

Steps, for every EPUB whose sha256 is not one of the books already in the corpus:
  1. read the sections in spine (reading) order, with the EPUB metadata
  2. drop what is not book text: cover / title / notice / about pages, navigation, index pages
     that are only listings, endnote sections, page headers, page numbers, running headers,
     store / social links, wiki markup, footnote reference numbers
  3. measure the extraction quality (Unicode, presentation forms, OCR letter confusions,
     reversed lines, broken words, share of known words); OCR pages that are mostly noise are
     excluded, with the reason
  4. duplicate statistics inside the batch, and overlap with dataset v2 (general Arabic, the
     earlier books, After-Sales)
  5. tokenizer statistics with the official old BPE 4K (no retraining)

The book text is never rephrased. The only character changes are Unicode normalizations that are
recorded per book: presentation forms -> normal letters (NFKC on those characters only) and the
Persian / Urdu letter shapes that OCR puts in Arabic text (ک ڪ ی ھ ۰-۹) -> Arabic letters.
Outputs (git-ignored, private):
    data/processed/private_books_batch2_extracted/   every book after extraction, before quality filtering
    data/processed/private_books_v2_candidate/       the books and sections that pass, with provenance.json
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from data.clean import clean_text
from data.extract_repo import html_to_text, looks_reversed
from data.sources import PRIVATE_SOURCES

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------
ARABIC_LETTER = re.compile(r"[ء-غف-يٱ-ۓ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
PRESENTATION_FORM = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
DIACRITIC = re.compile(r"[ً-ْٰ]")
EASTERN_DIGIT = re.compile(r"[٠-٩]")
PERSIAN_DIGIT = re.compile(r"[۰-۹]")
# letter shapes that do not belong in Arabic text; OCR of Arabic print produces them
OCR_LETTERS = {"ک": "ك", "ڪ": "ك", "ی": "ي", "ھ": "ه", **{chr(0x06F0 + i): chr(0x0660 + i) for i in range(10)}}
OCR_OTHER = re.compile(r"[گپچژٹڈڑںہےۀۃ]")  # Persian/Urdu letters with no Arabic counterpart
MARKS = re.compile(r"[ً-ْٰـ]")
AR_WORD = re.compile(r"[ء-يٱ]+")
WORD = re.compile(r"[^\W\d_]+")
SENTENCE_END = re.compile(r"[.!?؟]+(?=\s|$)")

# ---------------------------------------------------------------------------
# Lines and sections that are not book text
# ---------------------------------------------------------------------------
FRONT_MATTER_HREF = re.compile(r"(^|/)(cover|title|titlepage|notice|nav|toc|about|copyright|fp\d+)\.x?html$", re.I)
IA_ACCURACY = re.compile(r"The text on this page is estimated to be only ([\d.]+)% accurate")
PAGE_NUMBER = re.compile(r"^[\s\d٠-٩۰-۹\-–—.()\[\]{}|/]*$")
LINK = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+|(?:Twitter|twitter|تويتر|Telegram|تلغرام|تيليجرام)\s*:?\s*@?\w+|@\w{3,}")
WIKI_MARKUP = re.compile(r"\[\[[^\]]*\]\]|نزل نسخة مطبوعة|اقرأ عن .{1,60} في ويكيبيديا، الموسوعة الحرة")
NOTEREF = re.compile(r'<a[^>]*class="noteref"[^>]*>.*?</a>|<sup[^>]*>\s*\[?\d+\]?\s*</sup>', re.S)
ENDNOTE_START = re.compile(r"^\s*\[\s*←\s*\d+\s*\]")


def strip_marks(word):
    return MARKS.sub("", word)


# ---------------------------------------------------------------------------
# EPUB reading
# ---------------------------------------------------------------------------

def read_epub_sections(path):
    """(metadata, [(href, html)]) in spine order. Spine entries missing from the manifest are skipped."""
    with zipfile.ZipFile(path) as zf:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        opf_path = next(el.get("full-path") for el in container.iter() if el.tag.endswith("rootfile"))
        opf = ET.fromstring(zf.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        meta = {}
        for el in opf.iter():
            tag = el.tag.split("}")[-1]
            if tag in ("title", "creator", "language", "publisher", "rights", "date", "contributor", "subject") \
                    and (el.text or "").strip():
                meta.setdefault(tag, []).append(el.text.strip())
        manifest = {el.get("id"): el.get("href") for el in opf.iter() if el.tag.endswith("}item")}
        spine = [manifest[el.get("idref")] for el in opf.iter()
                 if el.tag.endswith("}itemref") and el.get("idref") in manifest]
        sections = []
        for href in spine:
            try:
                sections.append((href, zf.read(base + href).decode("utf-8", "replace")))
            except KeyError:
                continue
    return meta, sections


def detect_kind(meta, sections):
    if any(h.endswith("notice.html") and "Internet Archive" in t for h, t in sections):
        return "internet_archive_ocr"
    if "Wikisource" in " ".join(meta.get("contributor", [])):
        return "wikisource"
    return "publisher_epub"


def section_to_text(html):
    """HTML -> text: no <head>, no image alt text, no footnote reference numbers."""
    html = re.sub(r"(?is)<head>.*?</head>", " ", html)
    html = re.sub(r'\salt="[^"]*"', "", html)
    html = NOTEREF.sub("", html)
    html = re.sub(r"\s*\n\s*", " ", html)  # inside HTML a raw line break is a space
    return html_to_text(html)


# ---------------------------------------------------------------------------
# Normalization (recorded, character level only)
# ---------------------------------------------------------------------------

def normalize_chars(text, counts):
    counts["presentation_forms"] += len(PRESENTATION_FORM.findall(text))
    text = PRESENTATION_FORM.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    for src, dst in OCR_LETTERS.items():
        n = text.count(src)
        if n:
            counts[f"{src}->{dst}"] += n
            text = text.replace(src, dst)
    counts["bidi_and_invisible_marks"] += len(re.findall(r"[​-‏‪-‮⁦-⁩﻿­]", text))
    return clean_text(text)


# ---------------------------------------------------------------------------
# Quality measures
# ---------------------------------------------------------------------------
PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك")
SUFFIXES = ("هما", "كما", "هم", "هن", "كم", "نا", "ها", "ه", "ي", "ك", "ة", "ات", "ون", "ين", "ان")


class Lexicon:
    """Known Arabic words (from dataset v2), with light prefix/suffix stripping."""

    def __init__(self, texts):
        self.words = Counter(strip_marks(w) for t in texts for w in AR_WORD.findall(t))
        self.known = lru_cache(maxsize=None)(self._known)

    def _known(self, word):
        word = strip_marks(word)
        if word in self.words:
            return True
        for p in PREFIXES:
            if word.startswith(p) and len(word) - len(p) >= 2:
                rest = word[len(p):]
                if rest in self.words or any(rest.endswith(s) and rest[: -len(s)] in self.words for s in SUFFIXES):
                    return True
        return any(word.endswith(s) and len(word) - len(s) >= 2 and word[: -len(s)] in self.words for s in SUFFIXES)


def quality(text, lexicon):
    letters = ARABIC_LETTER.findall(text)
    latin = LATIN_LETTER.findall(text)
    words = AR_WORD.findall(text)
    lines = [l for l in text.split("\n") if l.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n|\n", text) if WORD.search(p)]
    sentence_words = [len(WORD.findall(s)) for p in paragraphs for s in SENTENCE_END.split(p) if WORD.search(s)]
    n_letters = len(letters) + len(latin)
    # OCR error markers: "أمامهء" (a comma read as hamza), "»" without "«", short Latin noise in Arabic lines
    hamza_comma = sum(w.endswith("ء") and len(w) > 2 and not lexicon.known(w) and lexicon.known(w[:-1]) for w in words)
    unmatched = max(0, text.count("»") - text.count("«"))
    latin_noise = sum(len(re.findall(r"\b[A-Za-z]{1,3}\b", l)) for l in lines
                      if len(ARABIC_LETTER.findall(l)) > 4 * len(LATIN_LETTER.findall(l)))
    per_1000 = 1000 / max(len(words), 1)
    return {
        "chars": len(text),
        "words": len(WORD.findall(text)),
        "arabic_letter_share": round(len(letters) / max(n_letters, 1), 4),
        "latin_letter_share": round(len(latin) / max(n_letters, 1), 4),
        "known_word_rate": round(sum(lexicon.known(w) for w in words) / max(len(words), 1), 4),
        "single_letter_word_rate": round(sum(len(strip_marks(w)) == 1 and w not in "وبلفك" for w in words) / max(len(words), 1), 4),
        "detached_article_rate": round(text.count(" ال ") / max(len(words), 1), 5),  # "ال جنسية": a broken word
        "other_persian_letters": len(OCR_OTHER.findall(text)),
        "hamza_as_comma_per_1000w": round(hamza_comma * per_1000, 2),
        "unmatched_closing_guillemets_per_1000w": round(unmatched * per_1000, 2),
        "latin_noise_tokens_per_1000w": round(latin_noise * per_1000, 2),
        "sentence_ends_per_100w": round(100 * len(SENTENCE_END.findall(text)) / max(len(WORD.findall(text)), 1), 2),
        "reversed_lines": sum(bool(looks_reversed(l)) for l in lines),
        "diacritics_per_100_letters": round(100 * len(DIACRITIC.findall(text)) / max(len(letters), 1), 2),
        "paragraphs": len(paragraphs),
        "avg_paragraph_words": round(statistics.mean(len(WORD.findall(p)) for p in paragraphs), 1) if paragraphs else 0,
        "avg_sentence_words": round(statistics.mean(sentence_words), 1) if sentence_words else 0,
    }


def grade(book):
    """Extraction quality grade from the measures above (thresholds chosen on this batch, see README)."""
    q = book["quality"]
    if not q or q["chars"] < 5000:
        return "F: (almost) nothing usable"
    confusions = sum(n for k, n in book["cleaning_counts"].items() if "->" in k)
    noise = (1000 * (confusions + q["other_persian_letters"]) / max(q["words"], 1) + q["hamza_as_comma_per_1000w"]
             + q["unmatched_closing_guillemets_per_1000w"] + q["latin_noise_tokens_per_1000w"])
    book["ocr_noise_per_1000w"] = round(noise, 1)
    if noise >= 25 or q["known_word_rate"] < 0.7:
        return "C: heavy OCR noise"
    if noise >= 5:
        return "B: OCR, readable with systematic errors"
    if q["sentence_ends_per_100w"] < 1:
        return "B: clean characters, but almost no punctuation"
    return "A: clean"


def is_listing(text):
    """An index / table-of-contents page: almost only short lines without sentence punctuation."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    short = sum(len(l.split()) <= 8 and not SENTENCE_END.search(l) for l in lines)
    return short / len(lines) >= 0.8 and statistics.median(len(l.split()) for l in lines) <= 6


# ---------------------------------------------------------------------------
# One book
# ---------------------------------------------------------------------------

def extract_book(path, lexicon, min_known=0.55, min_ia_accuracy=60.0):
    meta, sections = read_epub_sections(path)
    kind = detect_kind(meta, sections)
    title = (meta.get("title") or [""])[0]
    counts = Counter()
    used, excluded = [], []

    pages = []
    for href, html in sections:
        accuracy = IA_ACCURACY.search(html)
        text = section_to_text(html)
        text = IA_ACCURACY.sub("", text)
        pages.append({"href": href, "accuracy": float(accuracy.group(1)) if accuracy else None, "text": text})

    # running headers / footers: short lines repeated in many sections
    line_sections = Counter()
    for p in pages:
        for l in {" ".join(l.split()) for l in p["text"].split("\n") if l.strip()}:
            if len(l.split()) <= 8:
                line_sections[l] += 1
    repeated = {l for l, n in line_sections.items() if n >= max(5, len(pages) // 20)}

    for p in pages:
        href, text = p["href"], p["text"]
        if FRONT_MATTER_HREF.search(href):
            excluded.append({"section": href, "reason": "front/back matter (cover, title, notice, navigation, about)",
                             "chars": len(text.strip())})
            continue
        kept = []
        for line in text.split("\n"):
            key = " ".join(line.split())
            if not key:
                kept.append("")
                continue
            if re.fullmatch(r"Page \d+", key) or PAGE_NUMBER.match(key):
                counts["page_header_or_number_lines"] += 1
                continue
            if key in repeated or (title and key in (title, title.replace(" ", "_"))):
                counts["running_header_or_title_lines"] += 1
                continue
            new = WIKI_MARKUP.sub("", LINK.sub("", line))
            if new != line:
                counts["links_handles_wiki_markup_removed"] += 1
            kept.append(new)
        text = normalize_chars("\n".join(kept), counts)
        if not text.strip():
            excluded.append({"section": href, "reason": "empty after removing page headers / numbers", "chars": 0})
            continue
        if ENDNOTE_START.match(text):
            excluded.append({"section": href, "reason": "endnote (translator's note), detached from the text", "chars": len(text)})
            continue
        if is_listing(text):
            excluded.append({"section": href, "reason": "listing (table of contents / index of titles), not prose",
                             "chars": len(text)})
            continue
        q = quality(text, lexicon)
        reason = None
        if q["arabic_letter_share"] < 0.6:
            reason = f"not Arabic text ({q['arabic_letter_share']:.0%} Arabic letters): OCR noise or a foreign-language page"
        elif p["accuracy"] is not None and p["accuracy"] < min_ia_accuracy:
            reason = f"OCR page, Internet Archive estimates {p['accuracy']:.0f}% accuracy"
        elif q["known_word_rate"] < min_known and q["words"] >= 20:
            reason = f"OCR noise: only {q['known_word_rate']:.0%} of the words are known Arabic words"
        if reason:
            excluded.append({"section": href, "reason": reason, "chars": len(text)})
            continue
        used.append({"section": href, "text": text, "ia_accuracy": p["accuracy"]})

    text = "\n\n".join(u["text"] for u in used)
    return {
        "file": path.name, "kind": kind, "metadata": meta, "title": title,
        "author": (meta.get("creator") or [""])[0], "language": (meta.get("language") or [""])[0],
        "file_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sections_total": len(sections), "sections_used": [u["section"] for u in used],
        "sections_excluded": excluded, "cleaning_counts": dict(counts),
        "extracted_text": "\n\n".join(clean_text(p["text"]) for p in pages if not FRONT_MATTER_HREF.search(p["href"])),
        "text": text,
    }


# ---------------------------------------------------------------------------
# Duplicates and overlap
# ---------------------------------------------------------------------------

def norm_paragraph(p):
    return " ".join(strip_marks(p).split())


def paragraphs_of(text, min_words=5):
    return [p for p in (l.strip() for l in text.split("\n")) if len(p.split()) >= min_words]


def near_duplicates(paragraphs, threshold=0.8, n=5, bands=20, rows=3, seed=0):
    """Pairs of paragraphs whose word 5-gram sets have Jaccard >= threshold (MinHash LSH, then verified)."""
    rng = random.Random(seed)
    salts = [rng.getrandbits(64) for _ in range(bands * rows)]
    shingles = []
    for p in paragraphs:
        w = norm_paragraph(p).split()
        shingles.append({hash(" ".join(w[i:i + n])) for i in range(max(1, len(w) - n + 1))})
    buckets = {}
    for idx, s in enumerate(shingles):
        if len(s) < 4:
            continue
        sig = [min(h ^ salt for h in s) for salt in salts]
        for b in range(bands):
            buckets.setdefault((b, tuple(sig[b * rows:(b + 1) * rows])), []).append(idx)
    pairs = set()
    for ids in buckets.values():
        for i in range(len(ids)):
            for j in range(i + 1, min(len(ids), i + 50)):
                pairs.add((ids[i], ids[j]))
    result = []
    for i, j in pairs:
        a, b = shingles[i], shingles[j]
        jac = len(a & b) / len(a | b)
        if jac >= threshold and norm_paragraph(paragraphs[i]) != norm_paragraph(paragraphs[j]):
            result.append((i, j, round(jac, 3)))
    return result


def window_overlap(text, reference, window=50, step=50):
    windows = [text[i:i + window] for i in range(0, len(text) - window + 1, step)]
    return sum(w in reference for w in windows) / max(len(windows), 1)


def repeated_ngrams(text, n=8, top=5):
    w = norm_paragraph(text).split()
    counts = Counter(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
    return [(g, c) for g, c in counts.most_common(top) if c >= 3]


# ---------------------------------------------------------------------------
# Tokenizer statistics (official old BPE 4K)
# ---------------------------------------------------------------------------

def tokenizer_stats(tok, text, char_freq):
    ids = tok.encode(text)
    char_len = tok.token_char_lengths()
    byte_ids = [i for i in ids if i < 256]
    pieces = Counter(tok.token_text(i) for i in ids if i >= 256 and char_len[i] >= 2)
    return {
        "tokens": len(ids),
        "chars_per_token": round(len(text) / max(len(ids), 1), 3),
        "utf8_bytes_per_token": round(len(text.encode("utf-8")) / max(len(ids), 1), 3),
        "byte_fallback_token_rate": round(len(byte_ids) / max(len(ids), 1), 5),
        "byte_fallback_char_rate": round(sum(char_len[i] for i in byte_ids) / max(len(text), 1), 5),
        "fallback_characters": Counter(ch for ch in text if ch not in tok.char_to_id).most_common(10),
        "rare_char_rate": round(sum(char_freq.get(ch, 0) < 10 for ch in text if not ch.isspace()) / max(len(text), 1), 5),
        "eastern_arabic_digits": len(EASTERN_DIGIT.findall(text)),
        "persian_digits": len(PERSIAN_DIGIT.findall(text)),
        "top_subwords": [p for p, _ in pieces.most_common(15)],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def known_book_hashes():
    return {f["sha256"] for s in PRIVATE_SOURCES for f in s.get("files", [])}


def main():
    from tokenizer import load_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="folder with the EPUB files")
    parser.add_argument("--dataset", default="data/processed/hishamlm_v2")
    parser.add_argument("--tokenizer", default="data/processed/hishamlm_v1/tokenizers/bpe_4096.json")
    parser.add_argument("--extracted-out", default="data/processed/private_books_batch2_extracted")
    parser.add_argument("--out", default="data/processed/private_books_v2_candidate")
    parser.add_argument("--ids", default="", help="optional JSON file mapping file name -> short book id")
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    def read_all(name):
        return "\n".join((dataset / s / f"{name}.txt").read_text(encoding="utf-8") for s in ("train", "val", "test"))
    general_ar = "\n".join(read_all(n) for n in ("common_voice_ar", "ntrex_ar", "xquad_ar"))
    old_books = read_all("hindawi_books_ar")
    after_sales = read_all("after_sales")
    v2_all = "\n".join(p.read_text(encoding="utf-8") for p in sorted(dataset.glob("*/*.txt")))
    lexicon = Lexicon([general_ar, old_books])
    v2_paragraphs = {norm_paragraph(p) for p in paragraphs_of(v2_all)}
    tok = load_tokenizer(ROOT / args.tokenizer)
    char_freq = Counter((dataset / "train" / "hindawi_books_ar.txt").read_text(encoding="utf-8")) + Counter(general_ar)

    known = known_book_hashes()
    ids = json.loads(Path(args.ids).read_text(encoding="utf-8")) if args.ids else {}
    books = []
    for path in sorted(Path(args.raw).glob("*.epub")):
        if hashlib.sha256(path.read_bytes()).hexdigest() in known:
            print(f"skip (already in the corpus): {path.name}")
            continue
        book = extract_book(path, lexicon)
        book["id"] = ids.get(path.name, re.sub(r"\W+", "_", path.stem).strip("_").lower())
        books.append(book)
        print(f"{book['id']:28s} {book['kind']:22s} sections {len(book['sections_used']):4d}/{book['sections_total']:4d} "
              f"extracted {len(book['extracted_text']):>9,} kept {len(book['text']):>9,}", flush=True)

    # per-book measures
    all_paragraphs = []  # (book index, paragraph)
    for b_idx, book in enumerate(books):
        text = book["text"]
        book["quality_extracted"] = quality(book["extracted_text"], lexicon)
        book["quality"] = quality(text, lexicon) if text else None
        book["grade"] = grade(book)
        paras = paragraphs_of(text)
        all_paragraphs += [(b_idx, p) for p in paras]
        norm = Counter(norm_paragraph(p) for p in paras)
        book["duplicates"] = {
            "paragraphs": len(paras),
            "exact_duplicate_paragraphs": sum(c - 1 for c in norm.values() if c > 1),
            "repeated_8grams": repeated_ngrams(text) if text else [],
        }
        if text:
            book["overlap"] = {
                "general_arabic_50char": round(window_overlap(text, general_ar), 5),
                "earlier_books_50char": round(window_overlap(text, old_books), 5),
                "after_sales_50char": round(window_overlap(text, after_sales), 5),
                "v2_exact_paragraphs": sum(norm_paragraph(p) in v2_paragraphs for p in paras),
            }
            book["tokenizer"] = tokenizer_stats(tok, text, char_freq)

    # near duplicates across the whole batch
    near = near_duplicates([p for _, p in all_paragraphs])
    for book in books:
        book["duplicates"]["near_duplicate_pairs"] = 0
        book["duplicates"]["near_duplicate_pairs_cross_book"] = 0
    for i, j, _ in near:
        bi, bj = all_paragraphs[i][0], all_paragraphs[j][0]
        books[bi]["duplicates"]["near_duplicate_pairs"] += 1
        if bi != bj:
            books[bi]["duplicates"]["near_duplicate_pairs_cross_book"] += 1
            books[bj]["duplicates"]["near_duplicate_pairs_cross_book"] += 1
    books_of = defaultdict(set)
    for b, p in all_paragraphs:
        books_of[norm_paragraph(p)].add(b)
    cross_exact = sum(len(bs) > 1 for bs in books_of.values())

    # outputs
    ext_dir, out_dir = ROOT / args.extracted_out, ROOT / args.out
    ext_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "books").mkdir(parents=True, exist_ok=True)
    report = {"raw_dir": args.raw, "dataset_reference": args.dataset, "tokenizer": args.tokenizer,
              "near_duplicate_pairs_total": len(near), "cross_book_exact_duplicate_paragraphs": cross_exact,
              "examples_near_duplicates": [(all_paragraphs[i][1][:120], all_paragraphs[j][1][:120], jac)
                                           for i, j, jac in near[:10]],
              "books": []}
    for book in books:
        (ext_dir / f"{book['id']}.txt").write_text(book["extracted_text"], encoding="utf-8")
        if book["text"]:
            (out_dir / "books" / f"{book['id']}.txt").write_text(book["text"], encoding="utf-8")
        report["books"].append({k: v for k, v in book.items() if k not in ("text", "extracted_text")})
    provenance = [{
        "id": b["id"], "source_file": b["file"], "title": b["title"], "author": b["author"],
        "kind": b["kind"], "rights_in_metadata": b["metadata"].get("rights", []),
        "publisher_in_metadata": b["metadata"].get("publisher", []), "sha256": b["sha256"],
        "extracted_chars": len(b["extracted_text"]), "kept_chars": len(b["text"]),
        "sections_used": b["sections_used"], "sections_excluded": b["sections_excluded"],
        "cleaning_actions": b["cleaning_counts"]} for b in books]
    (out_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
