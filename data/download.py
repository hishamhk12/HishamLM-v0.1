"""Download the general sources listed in data/sources.py and record their provenance.

Usage (from the project root):
    python -m data.download

Files go to data/raw/general/<source>/ and every file's URL, license, size, sha256 and
download date are written to data/raw/general/provenance.json.
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from data.sources import SOURCES

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "general"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_path(source):
    return RAW_DIR / source["name"] / source["url"].rsplit("/", 1)[-1]


def download(source):
    path = source_path(source)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(source["url"], timeout=120) as response, open(tmp, "wb") as f:
            f.write(response.read())
        tmp.rename(path)
    digest = sha256_file(path)
    if source.get("sha256") and digest != source["sha256"]:
        raise ValueError(f"{source['name']}: sha256 {digest} does not match the pinned {source['sha256']}")
    return path, digest


def main():
    provenance_path = RAW_DIR / "provenance.json"
    old = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    provenance = {}
    for source in SOURCES:
        path, digest = download(source)
        previous = old.get(source["name"], {})
        provenance[source["name"]] = {
            "file": str(path.relative_to(ROOT)),
            "url": source["url"],
            "repository": source["repository"],
            "license": source["license"],
            "license_url": source["license_url"],
            "bytes": path.stat().st_size,
            "sha256": digest,
            # keep the first download date when the file is unchanged
            "downloaded_at": previous.get("downloaded_at") if previous.get("sha256") == digest
            else datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"{source['name']:16s} {path.stat().st_size:>10,} bytes  {digest[:12]}  {source['license']}")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
