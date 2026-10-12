"""Generate text from a trained checkpoint (MLP or Transformer), one token at a time.

Usage (from the project root):
    python -m inference.generate --checkpoint checkpoints/transformer_char/best.pt --prompt "The "
    python -m inference.generate --prompt "القط" --temperature 0      # greedy (always the most likely token)
"""

import argparse

import torch
import torch.nn.functional as F

from model.build import build_model
from tokenizer import tokenizer_from_dict
from tokenizer.char_tokenizer import CharTokenizer
from training.config import TrainConfig
from training.utils import get_device, set_seed


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = TrainConfig.from_dict(ckpt["config"])
    tokenizer = tokenizer_from_dict(ckpt["tokenizer"])
    model = build_model(cfg, tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, ckpt


@torch.no_grad()
def generate(model, ids, max_new_tokens, temperature=1.0, top_k=None, generator=None):
    """Append `max_new_tokens` tokens to `ids` (shape (1, T)).

    temperature == 0 -> greedy: always take the most likely token (deterministic).
    temperature > 0  -> sample from softmax(logits / temperature), optionally
                        restricted to the `top_k` most likely tokens.
    """
    for _ in range(max_new_tokens):
        context = ids[:, -model.context_length :]  # the model sees at most context_length tokens
        logits = model(context)[:, -1, :]  # prediction for the next token: (1, V)

        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).to(ids.device)

        ids = torch.cat([ids, next_id], dim=1)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/transformer_char/best.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    model, tokenizer, ckpt = load_checkpoint(args.checkpoint, device)

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt must contain at least one character")
    stoi = getattr(tokenizer, "stoi", None)  # BPE has byte fallback: nothing is unknown
    unknown = sorted({ch for ch in args.prompt if stoi is not None and ch not in stoi})
    if unknown:
        print(f"warning: characters not in vocabulary, mapped to {CharTokenizer.UNK_TOKEN}: {unknown}")

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = torch.Generator().manual_seed(args.seed)
    out = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, gen)

    print(f"{ckpt['model_type']} | checkpoint step {ckpt['step']} | val loss {ckpt['val_loss']:.4f}")
    print("-" * 40)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
