"""Pass 1 over Arabic Wikipedia XML chunks: keep the latest revision of every page.

Usage (from the project root):
    python -m data.wikipedia_extract data/wikipedia_ar/raw/arwiki-2026-08-01-p1007094p1086069.xml.bz2 [...]

The chunks are full-history exports (every revision of every page, all namespaces). They are read as a
stream (bzcat in a separate process, ~20x expansion), never decompressed to disk. Per input file it writes,
in data/processed/wikipedia_ar_work/ (git-ignored):
    <chunk>.pages.jsonl   one line per page: id, ns, title, redirect, revisions, latest revision id / timestamp, bytes
    <chunk>.ns0.jsonl     namespace-0 pages that are not redirects: latest revision wikitext (unchanged)
"""

import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/wikipedia_ar_work"
REDIRECT_TEXT = re.compile(r"^\s*#\s*(تحويل|REDIRECT)", re.IGNORECASE)
TAG_VALUE = re.compile(r"<(\w+)>(.*)</\1>")


def pages(lines):
    """Yield one dict per <page>, holding the text of its last <revision> only."""
    page = None
    in_revision = in_text = False
    text_lines = []
    for line in lines:
        if in_text:
            end = line.find("</text>")
            if end >= 0:
                text_lines.append(line[:end])
                page["text"] = "".join(text_lines)
                in_text = False
            else:
                text_lines.append(line)
            continue
        s = line.strip()
        if s == "<page>":
            page = {"title": None, "ns": None, "id": None, "redirect": None, "revisions": 0, "text": ""}
        elif page is None:
            continue
        elif s == "<revision>":
            in_revision = True
            page["revisions"] += 1
        elif s == "</revision>":
            in_revision = False
        elif s == "</page>":
            yield page
            page = None
        elif s.startswith("<text"):
            start = line.find(">") + 1
            if s.endswith("/>"):  # empty or deleted revision
                page["text"] = ""
                continue
            end = line.find("</text>", start)
            if end >= 0:
                page["text"] = line[start:end]
            else:
                text_lines, in_text = [line[start:]], True
        elif s == "<contributor>":
            page["_contributor"] = True
        elif s == "</contributor>":
            page["_contributor"] = False
        elif s.startswith("<redirect"):
            m = re.search(r'title="([^"]*)"', s)
            page["redirect"] = html.unescape(m.group(1)) if m else ""
        else:
            m = TAG_VALUE.match(s)
            if not m:
                continue
            tag, value = m.groups()
            if not in_revision and tag in ("title", "ns", "id") and page[tag] is None:
                page[tag] = html.unescape(value) if tag == "title" else int(value)
            elif in_revision and tag == "id" and not page.get("_contributor"):
                page["rev_id"] = int(value)
            elif in_revision and tag == "timestamp":
                page["timestamp"] = value


def extract(path):
    path = Path(path)
    chunk = path.name.removesuffix(".xml.bz2")
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["bzcat", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20)
    lines = (b.decode("utf-8", "replace") for b in proc.stdout)
    start, n, kept = time.time(), 0, 0
    with open(OUT / f"{chunk}.pages.jsonl", "w", encoding="utf-8") as meta, \
         open(OUT / f"{chunk}.ns0.jsonl", "w", encoding="utf-8") as articles:
        for page in pages(lines):
            n += 1
            text = html.unescape(page["text"])
            redirect = page["redirect"] is not None or bool(REDIRECT_TEXT.match(text))
            row = {"id": page["id"], "ns": page["ns"], "title": page["title"], "redirect": redirect,
                   "revisions": page["revisions"], "rev_id": page.get("rev_id"), "timestamp": page.get("timestamp"),
                   "bytes": len(text.encode("utf-8"))}
            meta.write(json.dumps(row, ensure_ascii=False) + "\n")
            if page["ns"] == 0 and not redirect and text.strip():
                articles.write(json.dumps({**row, "chunk": chunk, "wikitext": text}, ensure_ascii=False) + "\n")
                kept += 1
            if n % 20000 == 0:
                print(f"{chunk}: {n:,} pages, {kept:,} articles, {time.time() - start:.0f}s", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"bzcat failed on {path}")
    print(f"{chunk}: done, {n:,} pages, {kept:,} articles kept, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        extract(p)
