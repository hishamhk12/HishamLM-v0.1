"""Byte-Pair Encoding (BPE) tokenizer, written from scratch.

Vocabulary layout (token ids):
    0 .. 255            the 256 possible bytes   (byte fallback for rare/unseen characters)
    256 .. 256+C-1      C base characters        (every character seen >= min_char_freq times in training text)
    256+C ..            merged tokens, in the order they were learned

How text becomes tokens:
  1. pre-tokenize: split text into words, numbers, punctuation and whitespace
     (a single leading space stays attached to the next word: " the", " التوصيل").
     Merges never cross these boundaries.
  2. each character becomes its base-character id; a character that is not a base
     character becomes the ids of its UTF-8 bytes (byte fallback), so nothing is <unk>.
  3. learned merges are applied in the order they were learned.

decode() joins the bytes of every token and decodes UTF-8, so decode(encode(text)) == text.
"""

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_MARKS = "ـً-ٰٟ"  # tatweel and Arabic diacritics belong to the word they decorate
PRETOKENIZE = re.compile(
    rf" ?(?:[^\W\d_]|[{_MARKS}])+"  # a word (letters + Arabic marks), with an optional leading space
    r"| ?\d"  # one digit at a time: numbers are not memorized as whole tokens
    rf"| ?(?:[^\w\s{_MARKS}]|_)+"  # punctuation / symbols
    r"|\s+"  # whitespace (newlines, repeated spaces)
)


def pretokenize(text):
    return PRETOKENIZE.findall(text)


class BPETokenizer:
    def __init__(self, chars, merges):
        self.chars = list(chars)
        self.merges = [tuple(m) for m in merges]
        self.char_to_id = {ch: 256 + i for i, ch in enumerate(self.chars)}

        # bytes of every token: what decode() joins together
        self.token_bytes = [bytes([b]) for b in range(256)] + [ch.encode("utf-8") for ch in self.chars]
        self.merge_rank, self.merge_result = {}, {}
        for rank, (a, b) in enumerate(self.merges):
            new_id = len(self.token_bytes)
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
            self.merge_rank[(a, b)] = rank
            self.merge_result[(a, b)] = new_id
        self._cache = {}

    # ---------------------------------------------------------------- training
    @classmethod
    def train(cls, texts, vocab_size, min_char_freq=10, weights=None, verbose=False):
        """Learn merges from `texts` (a list of training texts).

        weights: optional per-text weights. Word counts of text i are scaled so that the text
        contributes weights[i] of the total, e.g. the same mix the model is trained on.
        """
        weights = weights or [1.0] * len(texts)
        char_counts = Counter()
        word_counts = defaultdict(float)
        total_chars = sum(len(t) for t in texts)
        for text, w in zip(texts, weights):
            char_counts.update(text)
            scale = w * total_chars / max(len(text), 1) / sum(weights)
            for word, n in Counter(pretokenize(text)).items():
                word_counts[word] += n * scale

        chars = sorted(ch for ch, n in char_counts.items() if n >= min_char_freq)
        tok = cls(chars, [])
        n_merges = vocab_size - 256 - len(chars)
        if n_merges < 0:
            raise ValueError(f"vocab_size {vocab_size} is smaller than 256 bytes + {len(chars)} base characters")

        # every distinct word as a list of symbol ids, with its (weighted) count
        words = [tok._base_ids(w) for w in word_counts]
        freqs = list(word_counts.values())

        pair_counts = defaultdict(float)
        where = defaultdict(set)  # pair -> indices of words containing it
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                where[pair].add(wi)
        # max-heap of (count, pair); entries go stale when counts change and are skipped
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges, pieces = [], list(tok.token_bytes)  # pieces: bytes of every token so far (for progress output)
        while len(merges) < n_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0.0)
            if count < 1e-9 or -neg_count != count:
                continue  # stale entry
            new_id = 256 + len(chars) + len(merges)
            merges.append(pair)
            pieces.append(pieces[pair[0]] + pieces[pair[1]])
            changed = set()
            for wi in sorted(where.pop(pair)):
                old, f = words[wi], freqs[wi]
                new = _merge(old, pair, new_id)
                if new == old:
                    continue
                for p in zip(old, old[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                for p in zip(new, new[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    changed.add(p)
                for p in set(zip(old, old[1:])) - set(zip(new, new[1:])):
                    where[p].discard(wi)
                words[wi] = new
            pair_counts.pop(pair, None)
            for p in changed:
                if p != pair and pair_counts[p] > 1e-9:
                    heapq.heappush(heap, (-pair_counts[p], p))
            if verbose and len(merges) % 500 == 0:
                print(f"  {len(merges)} merges, last: {pieces[-1].decode('utf-8', 'replace')!r}")
        return cls(chars, merges)

    # ---------------------------------------------------------------- encode / decode
    def _base_ids(self, word):
        ids = []
        for ch in word:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            else:
                ids.extend(ch.encode("utf-8"))  # byte fallback
        return ids

    def _encode_word(self, word):
        if word in self._cache:
            return self._cache[word]
        ids = self._base_ids(word)
        while len(ids) > 1:
            # apply the earliest-learned merge available in this word (leftmost if repeated)
            rank, i = min((self.merge_rank.get(p, float("inf")), i) for i, p in enumerate(zip(ids, ids[1:])))
            if rank == float("inf"):
                break
            ids[i : i + 2] = [self.merge_result[(ids[i], ids[i + 1])]]
        self._cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for word in pretokenize(text):
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids):
        return b"".join(self.token_bytes[i] for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.token_bytes)

    def token_char_lengths(self):
        """Characters each token represents (a UTF-8 continuation byte counts 0, a lead byte 1)."""
        return [sum((b & 0xC0) != 0x80 for b in tb) for tb in self.token_bytes]

    def token_text(self, i):
        return self.token_bytes[i].decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- save / load
    def to_dict(self):
        return {"type": "bpe", "chars": self.chars, "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chars"], d["merges"])

    def save(self, path, extra=None):
        d = {**(extra or {}), **self.to_dict()}  # tokenizer data always wins over extra metadata
        Path(path).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids`, left to right, without overlaps."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
