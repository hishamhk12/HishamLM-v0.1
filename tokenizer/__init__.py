import json
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.char_tokenizer import CharTokenizer


def tokenizer_from_dict(d):
    """Rebuild a tokenizer saved with to_dict() (checkpoints store this)."""
    return BPETokenizer.from_dict(d) if d.get("type") == "bpe" else CharTokenizer.from_dict(d)


def load_tokenizer(path):
    return tokenizer_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
