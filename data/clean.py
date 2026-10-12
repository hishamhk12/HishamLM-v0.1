"""Conservative text cleaning shared by every corpus.

What it does:
  - removes the BOM and applies Unicode NFC (canonical composition only)
  - unifies line endings to "\\n"
  - removes control characters and invisible direction/format marks
  - collapses runs of spaces/tabs, trims line ends, keeps at most one blank line

What it deliberately does NOT do: change Arabic letters (no alef/yaa/taa-marbuta
unification), remove diacritics or tatweel, transliterate, or touch punctuation.
"""

import re
import unicodedata

# Invisible format characters that only carry layout/direction hints.
_INVISIBLE = dict.fromkeys(
    map(ord, "﻿​‎‏‪‫‬‭‮⁦⁧⁨⁩­"), None
)
_SPACES = re.compile(r"[^\S\n]+")  # any whitespace except newline (tabs, NBSP, thin spaces, ...)
_BLANK_LINES = re.compile(r"\n{3,}")


def _is_unwanted_control(ch):
    return unicodedata.category(ch) == "Cc" and ch not in "\n\t"


def clean_text(text):
    text = text.translate(_INVISIBLE)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if not _is_unwanted_control(ch))
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()
