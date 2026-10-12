"""Character-level tokenizer: every character is one token."""

import json
from pathlib import Path


class CharTokenizer:
    UNK_TOKEN = "<unk>"
    UNK_ID = 0

    def __init__(self, chars):
        # id 0 is reserved for unknown characters; real characters start at 1.
        self.itos = [self.UNK_TOKEN] + list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text):
        """Build the vocabulary from every distinct character in `text`."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(ch, self.UNK_ID) for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def token_char_lengths(self):
        """Characters each token represents: always 1 (including <unk>, which stands for one character)."""
        return [1] * self.vocab_size

    def to_dict(self):
        return {"chars": self.itos[1:]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chars"])

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
