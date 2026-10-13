"""Build the HishamLM dataset: general Arabic + general English + After-Sales domain.

Usage (from the project root):
    python -m data.download                    # general sources -> data/raw/general/
    python -m data.extract_repo --repo ...     # domain corpus  -> data/processed/domain_after_sales/
    python -m data.prepare                     # this script    -> data/processed/hishamlm_v1/

Steps:
  1. read every source into documents and clean them (data/clean.py)
  2. remove exact duplicate lines across all sources
  3. split without leakage:
       general: each document is cut into ~2,000-character blocks; whole blocks go to
                train / validation / test (90 / 5 / 5 of every source's characters)
       domain:  whole documents go to train / validation / test (80 / 10 / 10)
  4. build the character vocabulary from the TRAIN split only
  5. measure <unk> rates, baselines, and the planned sampling mix
  6. write {train,val,test}/<source>.txt, vocab.json and meta.json (with a dataset fingerprint)

Nothing is copied to change the mix: the domain weight is a sampling probability
(meta.json "sampling"), applied by the training data loader.
"""

import argparse
import hashlib
import json
import math
import random
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from data.clean import clean_text, dedup_lines
from data.download import PRIVATE_DIR, RAW_DIR, sha256_file, source_path
from data.extract_repo import html_to_text
from data.sources import PRIVATE_SOURCES, REJECTED, SOURCES
from data.stats import text_stats
from tokenizer.char_tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "after_sales"
SPLITS = ("train", "val", "test")


@dataclass
class PrepareConfig:
    out_dir: str = "data/processed/hishamlm_v1"
    include_private: bool = False  # add data.sources.PRIVATE_SOURCES (never redistribute the result)
    domain_dir: str = "data/processed/domain_after_sales/docs"
    eval_questions: str = "data/processed/eval_after_sales_questions/questions.jsonl"
    seed: int = 42
    block_chars: int = 2000
    general_val: float = 0.05
    general_test: float = 0.05
    domain_val: float = 0.10
    domain_test: float = 0.10
    domain_max_doc_share: float = 0.08  # larger domain documents always stay in train
    domain_share: float = 0.175  # planned share of training windows sampled from the domain


# ---------------------------------------------------------------------------
# Source readers: each returns a list of (doc_id, text) and the raw character count
# ---------------------------------------------------------------------------

_ARABIC_DIACRITICS = re.compile(r"[ً-ْ]")
_SENTENCE_END = re.compile(r"^[.!?؟…]|[.!?؟…»\"')]$")


def is_modern_arabic_sentence(line):
    """Keep prose sentences; drop Quranic verses, classical poetry and proverbs.

    Those lines are usually fully diacritized or have no sentence punctuation.
    """
    letters = sum(0x0621 <= ord(c) <= 0x064A for c in line) or 1
    return len(_ARABIC_DIACRITICS.findall(line)) / letters < 0.1 and bool(_SENTENCE_END.search(line))


_LEADING_PUNCT = re.compile(r"^([.!?؟…]+)\s*(.+?)$")


def fix_leading_punctuation(line):
    """Sentences typed right-to-left often get their final mark at the start: "! أمسكه" -> "أمسكه!"."""
    m = _LEADING_PUNCT.match(line)
    if m and not re.search(r"[.!?؟…]$", m.group(2)):
        return f"{m.group(2)}{m.group(1)}"
    return line


def sample_lines(lines, max_chars, seed):
    """A seeded random subset of lines (about max_chars long) in their original order."""
    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)
    chosen, total = [], 0
    for i in order:
        if total >= max_chars:
            break
        chosen.append(i)
        total += len(lines[i]) + 1
    return [lines[i] for i in sorted(chosen)]


def unwrap_paragraphs(text):
    """Gutenberg texts are hard-wrapped at ~70 columns: join each paragraph into one line."""
    paragraphs = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())


_EPUB_FRONT_MATTER = re.compile(r"(cover|fp\d+|copyright|toc|nav|titlepage)\.x?html$", re.IGNORECASE)


def read_epub(path, title, max_latin_share=0.4):
    """Book text in reading (spine) order, without front matter and reference/index sections.

    Skipped: cover, title pages, copyright page, table of contents, and any section whose letters
    are mostly Latin (notes, references, bibliographies, bilingual indexes). The book title,
    repeated as a header at the top of every chapter, is removed.
    """
    with zipfile.ZipFile(path) as zf:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        opf_path = next(el.get("full-path") for el in container.iter() if el.tag.endswith("rootfile"))
        opf = ET.fromstring(zf.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        manifest = {el.get("id"): el.get("href") for el in opf.iter() if el.tag.endswith("}item")}
        spine = [manifest[el.get("idref")] for el in opf.iter() if el.tag.endswith("}itemref")]
        sections, skipped = [], []
        for href in spine:
            if _EPUB_FRONT_MATTER.search(href):
                skipped.append(href)
                continue
            # inside HTML a raw line break is just a space; paragraphs come from the block tags
            text = html_to_text(re.sub(r"\s*\n\s*", " ", zf.read(base + href).decode("utf-8")))
            letters = [c for c in text if c.isalpha()]
            if letters and sum(c.isascii() for c in letters) / len(letters) >= max_latin_share:
                skipped.append(href)
                continue
            lines = [l for l in text.split("\n") if l.strip()]
            if lines and " ".join(lines[0].split()) == title:
                lines = lines[1:]
            sections.append("\n".join(lines))
    return "\n\n".join(sections), skipped


def read_source(source, seed):
    path = source_path(source)
    name = source["name"]
    if name == "hindawi_books_ar":
        docs, raw_chars = [], 0
        for entry in source["files"]:
            text, _ = read_epub(path / entry["file"], entry["title"])
            docs.append((entry["title"], text))
            raw_chars += len(text)
        return docs, raw_chars
    if name.startswith("common_voice"):
        raw = path.read_text(encoding="utf-8")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if source["language"] == "ar":
            lines = [fix_leading_punctuation(l) for l in lines if is_modern_arabic_sentence(l)]
        if source.get("max_chars"):
            lines = sample_lines(lines, source["max_chars"], f"{seed}:{name}")
        return [(name, "\n".join(lines))], len(raw)
    if name == "ntrex_ar":
        raw = path.read_text(encoding="utf-8")
        return [(name, raw)], len(raw)
    if name == "xquad_ar":
        data = json.loads(path.read_text(encoding="utf-8"))
        docs = [(a["title"], "\n".join(p["context"] for p in a["paragraphs"])) for a in data["data"]]
        return docs, sum(len(t) for _, t in docs)
    if name == "gutenberg_en":
        docs = []
        with zipfile.ZipFile(path) as zf:
            for book in source["books"]:
                docs.append((book, zf.read(f"gutenberg/{book}").decode("utf-8", errors="replace")))
        return [(d, unwrap_paragraphs(t)) for d, t in docs], sum(len(t) for _, t in docs)
    raise ValueError(f"no reader for source {name}")


def read_domain(domain_dir):
    files = sorted(Path(domain_dir).rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no domain documents in {domain_dir}; run data.extract_repo first")
    docs = [(f.name, f.read_text(encoding="utf-8")) for f in files]
    return docs, sum(len(t) for _, t in docs)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def make_blocks(text, block_chars):
    """Cut a document into consecutive blocks of about block_chars, only at line boundaries."""
    blocks, current, size = [], [], 0
    for line in text.split("\n"):
        current.append(line)
        size += len(line) + 1
        if size >= block_chars:
            blocks.append("\n".join(current).strip())
            current, size = [], 0
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def assign_splits(units, val_frac, test_frac, seed, eligible=None):
    """Shuffle units (id, chars) with a fixed seed; fill test, then val, up to their share of characters.

    Returns {unit_id: split}. Units not in `eligible` always go to train.
    """
    total = sum(chars for _, chars in units)
    order = list(units)
    random.Random(seed).shuffle(order)
    splits, filled = {}, {"test": 0, "val": 0}
    targets = {"test": test_frac * total, "val": val_frac * total}
    for unit_id, chars in order:
        split = "train"
        if eligible is None or unit_id in eligible:
            for name in ("test", "val"):
                if filled[name] < targets[name]:
                    split = name
                    filled[name] += chars
                    break
        splits[unit_id] = split
    return splits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def decontaminate(held_out, train, window=50, step=10):
    """Drop held-out lines that also occur in train, or share a `window`-character verbatim run with it.

    Exact line dedup misses sentences that reappear inside different, longer lines
    (common in the domain corpus, where several files render the same pages).
    """
    train_lines = {" ".join(l.split()) for l in train.split("\n") if l.strip()}
    kept, dropped = [], 0
    for line in held_out.split("\n"):
        key = " ".join(line.split())
        if key and (key in train_lines or any(line[i : i + window] in train for i in range(0, len(line) - window + 1, step))):
            dropped += 1
            continue
        kept.append(line)
    return clean_text("\n".join(kept)), dropped


def shared_window_rate(held_out, train, window=50, step=50):
    """Share of held-out `window`-character windows that appear verbatim in train (a leakage measure)."""
    windows = [held_out[i : i + window] for i in range(0, len(held_out) - window + 1, step)]
    return sum(w in train for w in windows) / max(len(windows), 1)


def unk_rate(text, tokenizer):
    return sum(ch not in tokenizer.stoi for ch in text) / max(len(text), 1)


def unigram_loss(train_counts, text, vocab_size):
    """Cross-entropy (nats/char) of `text` under add-one smoothed train character frequencies."""
    total = sum(train_counts.values()) + vocab_size
    counts = Counter(text)
    nll = sum(n * -math.log((train_counts.get(ch, 0) + 1) / total) for ch, n in counts.items())
    return nll / max(len(text), 1)


def lines_of(text, min_words=3):
    return {" ".join(l.split()) for l in text.split("\n") if len(l.split()) >= min_words}


def fingerprint(out_dir):
    h = hashlib.sha256()
    for path in sorted(Path(out_dir).rglob("*.txt")) + [Path(out_dir) / "vocab.json"]:
        h.update(str(path.relative_to(out_dir)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare(cfg, sources=SOURCES, verify_hashes=True):
    out_dir = ROOT / cfg.out_dir
    report = {}

    # ---- 1. read + clean ----
    docs = {}  # source -> [(doc_id, text)]
    for source in sources:
        path = source_path(source)
        files = [(path / f["file"], f["sha256"]) for f in source["files"]] if "files" in source else [(path, source.get("sha256"))]
        for file, expected in files:
            if verify_hashes and expected and sha256_file(file) != expected:
                raise ValueError(f"{file} does not match its pinned sha256; run python -m data.download")
        raw_docs, raw_chars = read_source(source, cfg.seed)
        docs[source["name"]] = [(d, clean_text(t)) for d, t in raw_docs]
        report[source["name"]] = {"language": source["language"], "eval_group": source.get("eval_group", "general"),
                                  "private": bool(source.get("private")),
                                  "file_bytes": sum(f.stat().st_size for f, _ in files),
                                  "chars_extracted": raw_chars,
                                  "chars_after_cleaning": sum(len(t) for _, t in docs[source["name"]])}
    domain_docs, domain_raw = read_domain(ROOT / cfg.domain_dir)
    docs[DOMAIN] = domain_docs
    report[DOMAIN] = {"language": "ar+en", "file_bytes": None, "chars_extracted": domain_raw,
                      "chars_after_cleaning": domain_raw}

    # ---- 2. exact line dedup across all sources (domain first, so domain lines are kept) ----
    seen = set()
    for name in [DOMAIN] + [s["name"] for s in sources]:
        texts, removed = dedup_lines([t for _, t in docs[name]], seen=seen)
        docs[name] = [(d, t) for (d, _), t in zip(docs[name], texts) if t]
        report[name]["duplicate_lines_removed"] = removed
        report[name]["chars_after_dedup"] = sum(len(t) for _, t in docs[name])

    # ---- 3. split ----
    split_texts = {split: {} for split in SPLITS}
    split_units = []  # metadata to reproduce the split
    for source in sources:
        name = source["name"]
        units = [(f"{d}#{i}", block) for d, text in docs[name] for i, block in enumerate(make_blocks(text, cfg.block_chars))]
        assignment = assign_splits([(u, len(b)) for u, b in units], cfg.general_val, cfg.general_test, f"{cfg.seed}:{name}")
        for split in SPLITS:
            split_texts[split][name] = "\n\n".join(b for u, b in units if assignment[u] == split)
        split_units += [{"source": name, "unit": u, "chars": len(b), "split": assignment[u]} for u, b in units]

    domain_total = sum(len(t) for _, t in docs[DOMAIN])
    eligible = {d for d, t in docs[DOMAIN] if len(t) <= cfg.domain_max_doc_share * domain_total}
    assignment = assign_splits([(d, len(t)) for d, t in docs[DOMAIN]], cfg.domain_val, cfg.domain_test,
                               f"{cfg.seed}:{DOMAIN}", eligible)
    if not {"val", "test"} <= set(assignment.values()):
        raise ValueError("no domain document could be held out; raise domain_max_doc_share")
    for split in SPLITS:
        split_texts[split][DOMAIN] = "\n\n".join(t for d, t in docs[DOMAIN] if assignment[d] == split)
    report[DOMAIN]["held_out_decontamination"] = {}
    for split in ("val", "test"):
        before = len(split_texts[split][DOMAIN])
        split_texts[split][DOMAIN], dropped = decontaminate(split_texts[split][DOMAIN], split_texts["train"][DOMAIN])
        report[DOMAIN]["held_out_decontamination"][split] = {
            "chars_before": before, "chars_after": len(split_texts[split][DOMAIN]), "lines_dropped": dropped}
        if not split_texts[split][DOMAIN]:
            raise ValueError(f"domain {split} set is empty after decontamination: held-out documents repeat train")
    split_units += [{"source": DOMAIN, "unit": d, "chars": len(t), "split": assignment[d]} for d, t in docs[DOMAIN]]

    # ---- 4. vocabulary from train only ----
    train_text = "\n\n".join(split_texts["train"].values())
    tokenizer = CharTokenizer.from_text(train_text)

    # ---- write files ----
    for split in SPLITS:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        for old in (out_dir / split).glob("*.txt"):
            old.unlink()
        for name, text in split_texts[split].items():
            (out_dir / split / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
    tokenizer.save(out_dir / "vocab.json")

    # ---- 5. metrics ----
    names = [s["name"] for s in sources]
    general = [s["name"] for s in sources if s.get("eval_group", "general") == "general"]
    groups = {  # evaluation sets A and B
        "A_general_val": "\n\n".join(split_texts["val"][n] for n in general),
        "A_general_test": "\n\n".join(split_texts["test"][n] for n in general),
        "B_domain_val": split_texts["val"][DOMAIN],
        "B_domain_test": split_texts["test"][DOMAIN],
    }
    # sources with their own eval_group (e.g. private books) get separate sets, so A keeps its v1 meaning
    for group in sorted({s["eval_group"] for s in sources if s.get("eval_group", "general") != "general"}):
        members = [s["name"] for s in sources if s.get("eval_group") == group]
        for split in ("val", "test"):
            groups[f"A_{group}_{split}"] = "\n\n".join(split_texts[split][n] for n in members)
    questions_path = ROOT / cfg.eval_questions
    questions = [json.loads(l)["question"] for l in questions_path.read_text(encoding="utf-8").splitlines()] \
        if questions_path.exists() else []

    train_counts = Counter(train_text)
    uniform = math.log(tokenizer.vocab_size)
    train_lines = lines_of(train_text)
    held_out_lines = set().union(*(lines_of(split_texts[s][n]) for s in ("val", "test") for n in split_texts[s]))

    for name in names + [DOMAIN]:
        r = report[name]
        stats = text_stats("\n".join(t for _, t in docs[name]))
        r.update({f"{split}_chars": len(split_texts[split][name]) for split in SPLITS})
        r["arabic_pct"], r["latin_pct"] = round(stats["arabic_pct"], 2), round(stats["latin_pct"], 2)
        r["unk_rate"] = {split: round(unk_rate(split_texts[split][name], tokenizer), 6) for split in SPLITS}

    train_chars = {n: len(split_texts["train"][n]) for n in names + [DOMAIN]}
    general_train = sum(train_chars[n] for n in names)
    weights = {n: (1 - cfg.domain_share) * train_chars[n] / general_train for n in names}
    weights[DOMAIN] = cfg.domain_share
    # Alternative mix: the general share split equally between Arabic and English,
    # then by size within each language.
    languages = sorted({s["language"] for s in sources})
    balanced = {DOMAIN: cfg.domain_share}
    for lang in languages:
        names = [s["name"] for s in sources if s["language"] == lang]
        lang_chars = sum(train_chars[n] for n in names)
        for n in names:
            balanced[n] = (1 - cfg.domain_share) / len(languages) * train_chars[n] / lang_chars
    natural_share = train_chars[DOMAIN] / sum(train_chars.values())

    all_text = "\n".join(t for n in docs for _, t in docs[n])
    all_stats = text_stats(all_text)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "sources": report,
        "provenance": {k: v for folder in ((RAW_DIR,) + ((PRIVATE_DIR,) if cfg.include_private else ()))
                       if (folder / "provenance.json").exists()
                       for k, v in json.loads((folder / "provenance.json").read_text(encoding="utf-8")).items()
                       if not k.split("/")[0] in {s["name"] for s in PRIVATE_SOURCES} or cfg.include_private},
        "licenses": {s["name"]: {k: s[k] for k in ("license", "license_url", "license_note", "repository", "description")}
                     for s in sources},
        "domain_metadata": "data/processed/domain_after_sales/metadata.json",
        "rejected_sources": REJECTED,
        "totals": {
            "chars": len(all_text),
            "arabic_pct": round(all_stats["arabic_pct"], 2),
            "latin_pct": round(all_stats["latin_pct"], 2),
            "arabic_share_of_letters": round(all_stats["arabic_share_of_letters"], 2),
            **{f"{split}_chars": sum(len(t) for t in split_texts[split].values()) for split in SPLITS},
        },
        "vocab_size": tokenizer.vocab_size,
        "eval_sets": {
            **{k: {"chars": len(v), "unk_rate": round(unk_rate(v, tokenizer), 6),
                   "uniform_baseline_loss": round(uniform, 4),
                   "unigram_baseline_loss": round(unigram_loss(train_counts, v, tokenizer.vocab_size), 4)}
               for k, v in groups.items()},
            "C_domain_questions": {"path": cfg.eval_questions, "questions": len(questions),
                                   "unk_rate": round(unk_rate("\n".join(questions), tokenizer), 6),
                                   "note": "held out for later instruction/chat evaluation; not used for LM loss or vocabulary"},
        },
        "sampling": {
            "weights": {n: round(w, 4) for n, w in weights.items()},
            "weights_language_balanced": {n: round(w, 4) for n, w in balanced.items()},
            "domain_share_planned": cfg.domain_share,
            "domain_share_natural": round(natural_share, 4),
            "domain_oversampling_factor": round(cfg.domain_share / natural_share, 2),
        },
        "leakage_checks": {
            "train_lines_also_in_val_or_test": len(train_lines & held_out_lines),
            "held_out_questions_in_train": sum(q in train_text for q in questions),
            # share of 50-character held-out windows found verbatim in the matching train text
            "shared_50char_windows": {
                "A_general": round(shared_window_rate(groups["A_general_val"] + groups["A_general_test"],
                                                      "\n".join(split_texts["train"][n] for n in general)), 4),
                **{g[: -len("_val")]: round(shared_window_rate(groups[g] + groups[g[: -len("_val")] + "_test"],
                                                               "\n".join(split_texts["train"][n] for n in names)), 4)
                   for g in groups if g.endswith("_val") and g not in ("A_general_val", "B_domain_val")},
                "B_domain": round(shared_window_rate(groups["B_domain_val"] + groups["B_domain_test"],
                                                     split_texts["train"][DOMAIN]), 4),
            },
        },
        "split_units": split_units,
    }
    meta["dataset_fingerprint"] = fingerprint(out_dir)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def print_report(meta):
    print(f"{'source':16s} {'lang':5s} {'extracted':>10s} {'cleaned':>10s} {'dedup':>10s} {'train':>10s} {'val':>8s} {'test':>8s}  AR%   EN%  license")
    for name, r in meta["sources"].items():
        lic = meta["licenses"].get(name, {}).get("license", "user's own repository")
        print(f"{name:16s} {r['language']:5s} {r['chars_extracted']:>10,} {r['chars_after_cleaning']:>10,} {r['chars_after_dedup']:>10,} "
              f"{r['train_chars']:>10,} {r['val_chars']:>8,} {r['test_chars']:>8,} {r['arabic_pct']:5.1f} {r['latin_pct']:5.1f}  {lic}")
    t = meta["totals"]
    print(f"total chars {t['chars']:,} | train {t['train_chars']:,} | val {t['val_chars']:,} | test {t['test_chars']:,}")
    print(f"arabic {t['arabic_pct']}% | latin {t['latin_pct']}% | arabic share of letters {t['arabic_share_of_letters']}%")
    print(f"vocab size (train only): {meta['vocab_size']}")
    for k, v in meta["eval_sets"].items():
        print(f"  {k}: {v}")
    print(f"sampling: {meta['sampling']}")
    print(f"leakage checks: {meta['leakage_checks']}")
    print(f"dataset fingerprint: {meta['dataset_fingerprint']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=PrepareConfig.out_dir)
    parser.add_argument("--seed", type=int, default=PrepareConfig.seed)
    parser.add_argument("--include-private", action="store_true",
                        help="also use data.sources.PRIVATE_SOURCES (personal use only, never redistribute)")
    args = parser.parse_args()
    if args.include_private and args.out_dir == PrepareConfig.out_dir:
        parser.error("--include-private needs its own --out-dir (the default is the public v1 dataset)")
    cfg = PrepareConfig(out_dir=args.out_dir, seed=args.seed, include_private=args.include_private)
    meta = prepare(cfg, SOURCES + PRIVATE_SOURCES if args.include_private else SOURCES)
    print_report(meta)


if __name__ == "__main__":
    main()
