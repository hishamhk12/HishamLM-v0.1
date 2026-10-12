"""Train a language model from random initialization.

Usage (from the project root):
    python -m training.train --config configs/tiny_char.json
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.mlp_lm import MLPLanguageModel
from tokenizer.char_tokenizer import CharTokenizer
from training.config import TrainConfig
from training.data import get_batch, load_text, train_val_split
from training.utils import get_device, set_seed

ROOT = Path(__file__).resolve().parents[1]


def compute_loss(model, x, y):
    logits = model(x)  # (B, T, V)
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), y.view(B * T))


@torch.no_grad()
def estimate_loss(model, data, cfg, device):
    # A fixed generator means every evaluation sees the same batches,
    # so losses at different steps are directly comparable.
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.batch_size, cfg.context_length, gen)
        losses.append(compute_loss(model, x.to(device), y.to(device)).item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, tokenizer, cfg, step, train_loss, val_loss):
    torch.save(
        {
            "model_type": "mlp",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def train(cfg):
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # ---- data ----
    text, files = load_text(ROOT / cfg.data_dir)
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(ids, cfg.val_fraction)

    print(f"files: {[f.name for f in files]}")
    print(f"characters: {len(text):,} | vocab size: {tokenizer.vocab_size}")
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    # ---- model ----
    model = MLPLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=cfg.context_length,
        embedding_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    print(f"device: {device} | parameters: {model.num_parameters():,}")
    print(f"expected loss of a uniform guess: ln({tokenizer.vocab_size}) = {math.log(tokenizer.vocab_size):.4f}")

    # ---- output files ----
    out_dir = ROOT / cfg.checkpoint_dir / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    tokenizer.save(out_dir / "vocab.json")
    log_file = open(out_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "train_loss", "val_loss"])

    def evaluate(step):
        train_loss = estimate_loss(model, train_data, cfg, device)
        val_loss = estimate_loss(model, val_data, cfg, device)
        print(f"step {step:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        log.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}"])
        log_file.flush()
        return train_loss, val_loss

    # ---- training loop ----
    batch_gen = torch.Generator().manual_seed(cfg.seed)
    evaluate(0)  # loss before any training
    start = time.time()

    for step in range(1, cfg.max_steps + 1):
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, batch_gen)
        x, y = x.to(device), y.to(device)

        loss = compute_loss(model, x, y)  # forward pass + loss
        loss.backward()  # backward pass: compute gradients
        optimizer.step()  # update the weights
        optimizer.zero_grad(set_to_none=True)  # reset gradients for the next step

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            train_loss, val_loss = evaluate(step)
            save_checkpoint(ckpt_path, model, optimizer, tokenizer, cfg, step, train_loss, val_loss)

    log_file.close()
    print(f"done in {time.time() - start:.1f}s | checkpoint: {ckpt_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_char.json")
    args = parser.parse_args()
    train(TrainConfig.from_json(args.config))


if __name__ == "__main__":
    main()
