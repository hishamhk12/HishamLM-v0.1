"""Extract a clean domain text corpus from a local copy of a repository.

The repository is only read, never modified.

Usage (from the project root):
    python -m data.extract_repo --repo /path/to/after-seals --out data/processed/domain_after_sales

For every file the extractor decides one of:
  used      text was extracted and kept
  excluded  with a reason (dependency dir, binary/media, generated, possible secret, ...)

Outputs in --out:
  docs/NNN_<name>.txt   cleaned, redacted, deduplicated text of each used file
  corpus.txt            all docs joined
  metadata.json         per-file source path, type, status, reason and character counts
"""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from data.clean import clean_text
from data.stats import is_arabic, text_stats

# ---------------------------------------------------------------------------
# File selection rules
# ---------------------------------------------------------------------------

SKIP_DIRS = {".git"}  # not part of the project's files; not counted
DEPENDENCY_DIRS = {"node_modules", "vendor", "bower_components", ".venv", "venv", "__pycache__"}
BUILD_DIRS = {"dist", "build", "out", ".next", ".nuxt", ".cache", "cache", "coverage", ".vercel", ".turbo"}

MEDIA_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg", ".tif", ".tiff",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".ogg", ".m4a",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
}
BINARY_EXT = {".exe", ".dll", ".so", ".dylib", ".bin", ".zip", ".gz", ".tar", ".7z", ".rar", ".jar", ".class", ".pyc"}
LOCK_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock", "composer.lock", "Cargo.lock"}
CONFIG_FILES = {"package.json", "vercel.json", "tsconfig.json", ".gitignore", ".vercelignore", ".npmrc", ".editorconfig"}
STYLE_EXT = {".css", ".scss", ".less"}
SCRIPT_EXT = {".ps1", ".cmd", ".bat", ".sh"}

# Files whose name suggests credentials or raw records (customers, logs, exports).
SENSITIVE_NAME = re.compile(
    r"(^\.env|\.pem$|\.key$|\.p12$|id_rsa|credential|secret|\.log$|\.sql$|\.sqlite$|\.db$|dump|\.csv$|\.xlsx?$)",
    re.IGNORECASE,
)
# Evaluation sets and test code: kept out of training so they can be used as held-out checks later.
EVAL_OR_TEST_NAME = re.compile(r"(evaluat|dataset|^test|^validate|regression|\.test\.|\.spec\.)", re.IGNORECASE)

TEXT_EXT = {".md", ".txt", ".rst"}
HTML_EXT = {".html", ".htm"}
CODE_EXT = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
PPTX_EXT = {".pptx"}

# ---------------------------------------------------------------------------
# Secrets and personal data
# ---------------------------------------------------------------------------

SECRET_PATTERNS = re.compile(
    r"((?<![\w-])sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}"
    r"|(?:api[_-]?key|secret|password|passwd|access[_-]?token|auth[_-]?token)\s*[:=]\s*[\"'][^\"'\s]{8,}[\"'])",
    re.IGNORECASE,
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:(?:\+|00)966[\s-]?\d{8,9}|05\d{8})(?!\d)")
REDACTIONS = [  # applied in this order; the URL rule runs before the digit rule
    ("url", re.compile(r"(?:https?|file)://[^\s<>\"'`)\]]+"), "[URL]"),
    ("path", re.compile(r"\b[A-Za-z]:\\[^\s`\"'<>]*"), "[PATH]"),
    ("email", EMAIL, "[EMAIL]"),
    ("phone", PHONE, "[PHONE]"),
    ("long_number", re.compile(r"(?<!\d)\d{7,}(?!\d)"), "[ID]"),
]
MAX_PERSONAL_HITS = 3  # more emails/phones than this in one file -> treat as personal data and exclude


def redact(text):
    counts = {}
    for name, pattern, replacement in REDACTIONS:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = n
    return text, counts


# ---------------------------------------------------------------------------
# Domain relevance
# ---------------------------------------------------------------------------

DOMAIN_TERMS = [
    "خدمة", "خدمات", "العميل", "التوصيل", "التركيب", "الصيانة", "التصنيع", "القياسات", "المقاسات", "التصميم",
    "فاتورة", "الفاتورة", "مستودع", "المستودع", "شكوى", "الشكوى", "موعد", "فني", "الفني", "المهمة", "تذكرة",
    "odoo", "sap", "service", "customer", "delivery", "installation", "maintenance", "invoice", "warehouse",
    "ticket", "appointment", "technician", "after-sales", "complaint", "measurement", "manufacturing",
]
_DOMAIN_RE = re.compile("|".join(map(re.escape, DOMAIN_TERMS)), re.IGNORECASE)
MIN_DOMAIN_SCORE = 5.0  # domain-term hits per 1,000 characters


def domain_score(text):
    return 1000 * len(_DOMAIN_RE.findall(text)) / max(len(text), 1)


# ---------------------------------------------------------------------------
# Text extraction per file type
# ---------------------------------------------------------------------------

class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "li", "ul", "ol", "br", "tr", "td", "th", "table", "section", "article", "header",
             "footer", "h1", "h2", "h3", "h4", "h5", "h6", "figcaption", "figure", "blockquote", "summary", "details",
             "nav", "main", "aside", "dt", "dd", "label", "button", "option"}
    SKIP = {"script", "style", "svg", "noscript", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip_depth = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        if tag in self.BLOCK:
            self.parts.append("\n")
        alt = dict(attrs).get("alt")
        if alt and not self.skip_depth:
            self.parts.append(f"\n{alt}\n")  # image descriptions are useful text

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html):
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    return "".join(parser.parts)


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def markdown_to_text(md):
    lines, in_code = [], False
    for line in md.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code  # code blocks are not prose
            continue
        if in_code or _MD_TABLE_RULE.match(line) or line.strip() in {"---", "***", "___"}:
            continue
        line = _MD_LINK.sub(r"\1", line)
        line = re.sub(r"^\s*(#{1,6}|>)\s*", "", line)  # headings and quotes keep their text
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = line.strip().strip("|").replace(" | ", " — ") if line.strip().startswith("|") else line
        lines.append(line)
    return "\n".join(lines)


_JS_ESCAPES = {"n": "\n", "t": " ", "r": "", "b": "", "f": "", "v": "", "0": ""}


def js_string_literals(src):
    """Return the text of every string and template literal in JavaScript source, in source order.

    A small hand-written scanner: it skips comments, reads '...', "..." and `...`
    literals, and for `${expr}` inside templates scans expr for nested literals.
    It is best-effort (regex literals containing quotes can confuse it), which is
    fine because every literal is later filtered by is_prose().
    """
    out, n = [], len(src)

    def read_escape(i):  # src[i] == "\\"
        c = src[i + 1 : i + 2]
        if c == "u" and re.match(r"[0-9a-fA-F]{4}", src[i + 2 : i + 6]):
            return chr(int(src[i + 2 : i + 6], 16)), i + 6
        return _JS_ESCAPES.get(c, c), i + 2

    def read_quoted(i, quote):
        buf, i = [], i + 1
        while i < n:
            c = src[i]
            if c == "\\":
                ch, i = read_escape(i)
                buf.append(ch)
            elif c == quote:
                return "".join(buf), i + 1
            elif c == "\n":
                return None, i + 1  # not a real string literal
            else:
                buf.append(c)
                i += 1
        return None, n

    def read_template(i):
        slot = len(out)
        out.append(None)  # reserve the position so the output keeps source order
        buf, i = [], i + 1
        while i < n:
            c = src[i]
            if c == "\\":
                ch, i = read_escape(i)
                buf.append(ch)
            elif c == "`":
                i += 1
                break
            elif c == "$" and src[i + 1 : i + 2] == "{":
                start = len(out)
                i = scan(i + 2, inside_braces=True)
                inner = out[start:]
                # ${term("جديد", "New")} -> "جديد (New)": short plain strings are part of the sentence.
                if inner and all(s and len(s) <= 60 and "<" not in s for s in inner):
                    del out[start:]
                    buf.append(inner[0] + "".join(f" ({s})" for s in inner[1:]))
                else:
                    buf.append(" ")
            else:
                buf.append(c)
                i += 1
        out[slot] = "".join(buf)
        return i

    def scan(i, inside_braces=False):
        depth = 0
        while i < n:
            c = src[i]
            if c in "\"'":
                text, i = read_quoted(i, c)
                if text:
                    out.append(text)
            elif c == "`":
                i = read_template(i)
            elif src.startswith("//", i):
                j = src.find("\n", i)
                i = n if j < 0 else j
            elif src.startswith("/*", i):
                j = src.find("*/", i + 2)
                i = n if j < 0 else j + 2
            else:
                if inside_braces and c == "{":
                    depth += 1
                elif inside_braces and c == "}":
                    if depth == 0:
                        return i + 1
                    depth -= 1
                i += 1
        return i

    scan(0)
    return [s for s in out if s]


_CODE_HINT = re.compile(r"(=>|===|!==|&&|\|\||\bfunction\b|\breturn\b|\bconst\b|\blet\b|\bvar\(--|[{};]|#/|\.js\b|\.png\b|\bpx\b)")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9]+([-_][A-Za-z0-9]+)+$|^[a-z]+[A-Z][A-Za-z0-9]*$")


def is_prose(line):
    """Heuristic: does this line read like human text rather than code or CSS classes?"""
    words = line.split()
    if len(words) < 3 or len(line) < 15:
        return False
    if sum(ch.isalpha() for ch in line) / len(line) < 0.6 or _CODE_HINT.search(line):
        return False
    if not any(is_arabic(ch) for ch in line):
        if len(words) < 4 or sum(bool(_IDENTIFIER.match(w)) for w in words) > len(words) / 4:
            return False
    return True


def code_to_text(src):
    lines = []
    for literal in js_string_literals(src):
        text = html_to_text(literal) if "<" in literal and ">" in literal else literal
        lines.extend(line.strip() for line in text.split("\n") if is_prose(line.strip()))
    return "\n".join(lines)


def pptx_to_text(path):
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    with zipfile.ZipFile(path) as zf:
        def ordered(prefix):
            names = [n for n in zf.namelist() if re.fullmatch(rf"{prefix}\d+\.xml", n)]
            return sorted(names, key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))

        lines = []
        for name in ordered("ppt/slides/slide") + ordered("ppt/notesSlides/notesSlide"):
            root = ET.fromstring(zf.read(name))
            for para in root.iter(f"{ns}p"):
                text = "".join(t.text or "" for t in para.iter(f"{ns}t")).strip()
                if text and not text.isdigit():  # skip bare slide numbers
                    lines.append(text)
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classification and extraction
# ---------------------------------------------------------------------------

def classify(rel_path, path):
    """Return (kind, exclusion_reason). exclusion_reason is None for files to extract."""
    parts, name, ext = rel_path.parts, path.name, path.suffix.lower()
    if any(p in DEPENDENCY_DIRS for p in parts):
        return "dependency", "dependency directory"
    if any(p in BUILD_DIRS for p in parts):
        return "build", "build/cache directory"
    if SENSITIVE_NAME.search(name):
        return "sensitive", "file name suggests credentials or raw records"
    if ext in MEDIA_EXT:
        return "media", "image/video/audio/font"
    if ext in BINARY_EXT:
        return "binary", "binary file"
    if name in LOCK_FILES:
        return "lockfile", "lock file"
    if name in CONFIG_FILES:
        return "config", "configuration file"
    if ext in STYLE_EXT:
        return "style", "stylesheet"
    if ext in SCRIPT_EXT:
        return "script", "shell script"
    if ext == ".json":
        return "json", "structured/generated data (embeddings, evaluation outputs, extracted copies)"
    if ext == ".pdf":
        return "pdf", "PDF text extraction not supported yet"
    if ext in CODE_EXT:
        if ".min." in name:
            return "code", "minified file"
        if EVAL_OR_TEST_NAME.search(name):
            return "code", "evaluation set / test code (kept for future held-out evaluation)"
        return "code", None
    if ext in TEXT_EXT:
        return "markdown" if ext == ".md" else "text", None
    if ext in HTML_EXT:
        return "html", None
    if ext in PPTX_EXT:
        return "pptx", None
    return "other", "unsupported file type"


_MEDIA_REF = re.compile(r"\.(png|jpe?g|gif|webp|mp4|mov|webm)\b|\b\d{2}:\d{2}:\d{2}\b", re.IGNORECASE)


def is_media_listing(text, threshold=0.3):
    """Auto-generated indexes of images/video frames: many lines are file names or hh:mm:ss timestamps."""
    lines = [line for line in text.split("\n") if line.strip()]
    return bool(lines) and sum(bool(_MEDIA_REF.search(line)) for line in lines) / len(lines) > threshold


def looks_generated(src):
    head = src[:3000]
    if any(sig in head for sig in ("__commonJS", "webpackBootstrap", "__webpack_require__", "/*! For license")):
        return True
    lines = src.split("\n")
    return len(src) > 5000 and len(src) / max(len(lines), 1) > 500  # minified: very long lines


def extract(path, kind):
    """Return (text, raw_source_or_None). The raw source is used for the secret scan."""
    if kind == "pptx":
        return pptx_to_text(path), None
    src = path.read_text(encoding="utf-8-sig", errors="replace")
    if kind == "code":
        return code_to_text(src), src
    if kind == "html":
        return html_to_text(src), src
    if kind == "markdown":
        return markdown_to_text(src), src
    return src, src


def dedup_lines(docs, min_words=3):
    """Remove lines (with at least `min_words` words) already seen in an earlier document or line."""
    seen, removed = set(), 0
    result = []
    for text in docs:
        kept = []
        for line in text.split("\n"):
            key = " ".join(line.split())
            if len(key.split()) >= min_words:
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
            kept.append(line)
        result.append(clean_text("\n".join(kept)))
    return result, removed


def git_commit(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(repo, out_dir):
    repo, out_dir = Path(repo).resolve(), Path(out_dir)
    files = sorted(p for p in repo.rglob("*") if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(repo).parts))

    records, candidates = [], []
    for path in files:
        rel = path.relative_to(repo)
        kind, reason = classify(rel, path)
        rec = {"path": rel.as_posix(), "type": kind, "bytes": path.stat().st_size, "status": "excluded", "reason": reason}
        records.append(rec)
        if reason:
            continue

        text, raw = extract(path, kind)
        if raw is not None and kind == "code" and looks_generated(raw):
            rec["reason"] = "generated bundle or minified code"
            continue
        if SECRET_PATTERNS.search(raw if raw is not None else text):
            rec["reason"] = "possible secret (API key / token / password)"
            continue
        personal = len(EMAIL.findall(text)) + len(PHONE.findall(text))
        if personal > MAX_PERSONAL_HITS:
            rec["reason"] = f"possible personal data ({personal} emails/phone numbers)"
            continue

        text, redactions = redact(clean_text(text))
        text = clean_text(text)
        rec["extracted_chars"] = len(text)
        rec["redactions"] = redactions
        rec["domain_score"] = round(domain_score(text), 2)
        if len(text) < 200:
            rec["reason"] = "too little useful text extracted"
            continue
        if is_media_listing(text):
            rec["reason"] = "generated media/frame listing"
            continue
        if rec["domain_score"] < MIN_DOMAIN_SCORE:
            rec["reason"] = f"off-domain (domain score {rec['domain_score']} < {MIN_DOMAIN_SCORE})"
            continue
        candidates.append((rec, text))

    # ---- deduplication: identical files, then repeated lines across files ----
    chars_before = sum(len(t) for _, t in candidates)
    unique, seen_hashes = [], {}
    for rec, text in candidates:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            rec["reason"] = f"exact duplicate of {seen_hashes[h]}"
            continue
        seen_hashes[h] = rec["path"]
        unique.append((rec, text))
    deduped, lines_removed = dedup_lines([t for _, t in unique])

    # ---- write outputs ----
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for old in docs_dir.glob("*.txt"):
        old.unlink()
    used_texts = []
    for i, ((rec, _), text) in enumerate(zip(unique, deduped)):
        if len(text) < 200:
            rec["reason"] = "almost entirely duplicated by earlier files"
            continue
        out_name = f"{i:03d}_{re.sub(r'[^0-9A-Za-z؀-ۿ._-]+', '_', rec['path'])}.txt"
        (docs_dir / out_name).write_text(text + "\n", encoding="utf-8")
        rec.update(status="used", reason=None, chars_after_dedup=len(text), output=f"docs/{out_name}")
        used_texts.append(text)

    corpus = "\n\n".join(used_texts) + "\n"
    (out_dir / "corpus.txt").write_text(corpus, encoding="utf-8")

    stats = text_stats(corpus)
    used = [r for r in records if r["status"] == "used"]
    summary = {
        "files_scanned": len(records),
        "files_used": len(used),
        "files_excluded": len(records) - len(used),
        "chars_before_dedup": chars_before,
        "duplicate_files_removed": len(candidates) - len(unique),
        "duplicate_lines_removed": lines_removed,
        "chars_after_dedup": len(corpus),
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "arabic_pct": round(stats["arabic_pct"], 2),
        "latin_pct": round(stats["latin_pct"], 2),
        "arabic_share_of_letters": round(stats["arabic_share_of_letters"], 2),
        "used_by_type": dict(Counter(r["type"] for r in used)),
        "excluded_by_reason": dict(Counter(r["reason"] for r in records if r["status"] == "excluded")),
        "redactions": dict(sum((Counter(r.get("redactions", {})) for r in used), Counter())),
    }
    meta = {
        "source_repo": str(repo),
        "source_commit": git_commit(repo),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "files": records,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="path to a local clone (read only)")
    parser.add_argument("--out", default="data/processed/domain_after_sales")
    args = parser.parse_args()

    meta = run(args.repo, args.out)
    print(json.dumps(meta["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
