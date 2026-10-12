"""Generate text from a trained checkpoint, one token at a time.

Usage (from the project root):
    python -m inference.generate --checkpoint checkpoints/tiny_char/ckpt.pt --prompt "The "
"""

import argparse

import torch
import torch.nn.functional as F

from model.mlp_lm import MLPLanguageModel
from tokenizer.char_tokenizer import CharTokenizer
from training.utils import get_device, set_seed


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    model = MLPLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=cfg["context_length"],
        embedding_size=cfg["embedding_size"],
        hidden_size=cfg["hidden_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, ckpt


@torch.no_grad()
def generate(model, ids, max_new_tokens, temperature=1.0, top_k=None, generator=None):
    """Append `max_new_tokens` sampled tokens to `ids` (shape (1, T))."""
    for _ in range(max_new_tokens):
        context = ids[:, -model.context_length :]  # the model only needs the last window
        logits = model(context)[:, -1, :]  # prediction for the next token: (1, V)
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
    parser.add_argument("--checkpoint", default="checkpoints/tiny_char/ckpt.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
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
    unknown = sorted({ch for ch in args.prompt if ch not in tokenizer.stoi})
    if unknown:
        print(f"warning: characters not in vocabulary, mapped to {CharTokenizer.UNK_TOKEN}: {unknown}")

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = torch.Generator().manual_seed(args.seed)
    out = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, gen)

    print(f"checkpoint step {ckpt['step']} | val loss {ckpt['val_loss']:.4f}")
    print("-" * 40)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
