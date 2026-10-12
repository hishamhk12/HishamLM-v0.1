"""Train a BPE tokenizer on the TRAIN split of a prepared dataset.

Usage (from the project root):
    python -m tokenizer.train_bpe --vocab-size 2048

Each training source is weighted with the dataset's sampling weights, so the merges
follow the same Arabic / English / After-Sales mix the model is trained on.
Validation and test text are never read.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from tokenizer.bpe_tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/hishamlm_v1")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--sampling", default="weights_language_balanced")
    parser.add_argument("--min-char-freq", type=int, default=10)
    parser.add_argument("--out", help="default: <dataset>/tokenizers/bpe_<vocab-size>.json")
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    meta = json.loads((dataset / "meta.json").read_text(encoding="utf-8"))
    weights = meta["sampling"][args.sampling]
    names = sorted(weights)
    texts = [(dataset / "train" / f"{n}.txt").read_text(encoding="utf-8") for n in names]

    start = time.time()
    tok = BPETokenizer.train(texts, args.vocab_size, args.min_char_freq, [weights[n] for n in names], verbose=True)
    seconds = time.time() - start

    out = Path(args.out) if args.out else dataset / "tokenizers" / f"bpe_{args.vocab_size}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    chars_per_token = {}
    for n, text in zip(names, texts):
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"round trip failed on {n}"
        chars_per_token[n] = round(len(text) / len(ids), 3)
    extra = {"trained_on": {"dataset": args.dataset, "dataset_fingerprint": meta["dataset_fingerprint"],
                            "split": "train", "sampling": args.sampling, "weights": weights,
                            "min_char_freq": args.min_char_freq},
             "vocab_size": tok.vocab_size, "n_base_chars": len(tok.chars), "n_merges": len(tok.merges),
             "train_chars_per_token": chars_per_token, "training_seconds": round(seconds, 1)}
    tok.save(out, extra)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"vocab {tok.vocab_size} = 256 bytes + {len(tok.chars)} chars + {len(tok.merges)} merges | {seconds:.1f}s")
    print(f"chars per token on train: {chars_per_token}")
    print(f"saved {out} (sha256 {digest[:16]})")


if __name__ == "__main__":
    main()
