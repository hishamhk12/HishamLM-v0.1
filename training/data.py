"""Dataset pipeline: text -> token ids -> (input, target) batches.

Two ways to get data:
  - legacy (phases 1-2): every .txt in one folder, last part of the text is validation
  - prepared dataset (phase 3): data/processed/<name>/ from `python -m data.prepare`,
    several training sources mixed by sampling weights, separate evaluation sets
"""

import json
from pathlib import Path

import torch

from tokenizer.char_tokenizer import CharTokenizer

DOMAIN = "after_sales"


def load_text(raw_dir):
    """Read every .txt file directly in `raw_dir` (sorted by name) and join them."""
    files = sorted(Path(raw_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files found in {raw_dir}")
    return "\n".join(f.read_text(encoding="utf-8") for f in files), files


def train_val_split(ids, val_fraction):
    """Split a 1-D tensor of token ids: first part for training, the tail for validation."""
    n_train = int(len(ids) * (1 - val_fraction))
    return ids[:n_train], ids[n_train:]


def get_batch(data, batch_size, context_length, generator=None):
    """Sample `batch_size` random windows from `data`.

    x is a window of `context_length` tokens and y is the same window shifted
    one token to the right, so y[b, t] is the token that follows x[b, t].
    """
    if len(data) <= context_length:
        raise ValueError(f"data has {len(data)} tokens, need more than context_length={context_length}")
    starts = torch.randint(len(data) - context_length, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + context_length] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + context_length] for s in starts])
    return x, y


class TextData:
    """Training streams with sampling weights, plus named validation/test streams."""

    def __init__(self, tokenizer, train, train_weights, evals, val_weights, info=None):
        self.tokenizer = tokenizer
        self.train = train  # name -> 1-D LongTensor
        self.names = list(train)
        self.weights = torch.tensor([train_weights[n] for n in self.names], dtype=torch.float)
        self.evals = evals  # name -> 1-D LongTensor (validation and test sets)
        self.val_weights = val_weights  # which eval sets make up "val_loss", and how much each counts
        self.info = info or {}

    def train_batch(self, batch_size, context_length, generator=None):
        if len(self.names) == 1:
            return get_batch(self.train[self.names[0]], batch_size, context_length, generator)
        # Pick a source for every row according to the weights, then a random window inside it.
        # Weighting happens here; no text is duplicated on disk.
        picks = torch.multinomial(self.weights, batch_size, replacement=True, generator=generator)
        rows = [get_batch(self.train[self.names[p]], 1, context_length, generator) for p in picks.tolist()]
        return torch.cat([x for x, _ in rows]), torch.cat([y for _, y in rows])


def legacy_data(raw_dir, val_fraction, tokenizer=None):
    text, files = load_text(raw_dir)
    tokenizer = tokenizer or CharTokenizer.from_text(text)
    train, val = train_val_split(torch.tensor(tokenizer.encode(text), dtype=torch.long), val_fraction)
    info = {"files": [f.name for f in files], "characters": len(text)}
    return TextData(tokenizer, {"train": train}, {"train": 1.0}, {"val": val}, {"val": 1.0}, info)


def prepared_data(dataset_dir, sampling="weights"):
    """Load a dataset written by data.prepare.

    sampling: "weights" (general sources by size) or "weights_language_balanced".
    Evaluation sets: A = general (all general sources), B = After-Sales domain.
    """
    root = Path(dataset_dir)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    tokenizer = CharTokenizer.load(root / "vocab.json")

    def read(split, name):
        return (root / split / f"{name}.txt").read_text(encoding="utf-8")

    def encode(text):
        return torch.tensor(tokenizer.encode(text), dtype=torch.long)

    weights = meta["sampling"][sampling]
    general = [n for n in weights if n != DOMAIN]
    train = {n: encode(read("train", n)) for n in weights}
    evals = {}
    for split in ("val", "test"):
        evals[f"A_general_{split}"] = encode("\n\n".join(read(split, n) for n in general))
        evals[f"B_domain_{split}"] = encode(read(split, DOMAIN))
    # Checkpoints are chosen on a validation loss mixed like the training data.
    val_weights = {"A_general_val": 1 - weights[DOMAIN], "B_domain_val": weights[DOMAIN]}
    info = {"dataset": str(dataset_dir), "fingerprint": meta["dataset_fingerprint"], "sampling": sampling,
            "weights": weights, "train_tokens": {n: len(t) for n, t in train.items()},
            "eval_tokens": {n: len(t) for n, t in evals.items()}, "baselines": meta["eval_sets"]}
    return TextData(tokenizer, train, weights, evals, val_weights, info)
