"""Download the general sources listed in data/sources.py and record their provenance.

Usage (from the project root):
    python -m data.download

Files go to data/raw/general/<source>/ and every file's URL, license, size, sha256 and
download date are written to data/raw/general/provenance.json.

    python -m data.download --include-private

also fetches the private sources (data.sources.PRIVATE_SOURCES) into data/raw/private_books/,
with their own provenance.json. Both folders are ignored by git.
"""

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from data.sources import PRIVATE_SOURCES, SOURCES

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "general"
PRIVATE_DIR = ROOT / "data" / "raw" / "private_books"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_path(source):
    """The downloaded file of a single-file source, or the folder of a multi-file source."""
    base = (PRIVATE_DIR if source.get("private") else RAW_DIR) / source["name"]
    return base if "files" in source else base / source["url"].rsplit("/", 1)[-1]


def fetch(url, path, expected_sha256, label):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as f:
            f.write(response.read())
        tmp.rename(path)
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{label}: sha256 {digest} does not match the pinned {expected_sha256}")
    return digest


def download(source):
    """Download a source and return [(path, sha256, file entry or None), ...]."""
    if "files" in source:
        folder = source_path(source)
        return [(folder / f["file"], fetch(f["url"], folder / f["file"], f["sha256"], f"{source['name']}/{f['file']}"), f)
                for f in source["files"]]
    path = source_path(source)
    return [(path, fetch(source["url"], path, source.get("sha256"), source["name"]), None)]


def write_provenance(sources, folder):
    provenance_path = folder / "provenance.json"
    old = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    provenance = {}
    for source in sources:
        for path, digest, entry in download(source):
            key = source["name"] if entry is None else f"{source['name']}/{entry['file']}"
            previous = old.get(key, {})
            provenance[key] = {
                "file": str(path.relative_to(ROOT)),
                "url": (entry or source)["url"],
                "repository": source["repository"],
                "license": source["license"],
                "license_url": source["license_url"],
                **({k: entry[k] for k in ("title", "author", "rights")} if entry else {}),
                "bytes": path.stat().st_size,
                "sha256": digest,
                # keep the first download date when the file is unchanged
                "downloaded_at": previous.get("downloaded_at") if previous.get("sha256") == digest
                else datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            print(f"{key:40s} {path.stat().st_size:>10,} bytes  {digest[:12]}  {source['license'][:40]}")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-private", action="store_true", help="also fetch data.sources.PRIVATE_SOURCES")
    args = parser.parse_args()
    write_provenance(SOURCES, RAW_DIR)
    if args.include_private:
        write_provenance(PRIVATE_SOURCES, PRIVATE_DIR)


if __name__ == "__main__":
    main()
