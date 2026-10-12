"""Dataset pipeline: raw .txt files -> one text -> token ids -> (input, target) batches."""

from pathlib import Path

import torch


def load_text(raw_dir):
    """Read every .txt file in `raw_dir` (sorted by name) and join them."""
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
