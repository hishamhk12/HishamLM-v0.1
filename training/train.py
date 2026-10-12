"""Train a language model (MLP or Transformer) from random initialization.

Usage (from the project root):
    python -m training.train --model transformer          # uses configs/transformer_char.json
    python -m training.train --model mlp                  # uses configs/mlp_char.json
    python -m training.train --config configs/my_run.json
    python -m training.train --model transformer --resume --max-steps 4000
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.build import MODEL_TYPES, build_model
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


def train(cfg, resume=False):
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    out_dir = ROOT / cfg.checkpoint_dir / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = out_dir / "latest.pt", out_dir / "best.pt"

    ckpt = None
    if resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"cannot resume: {latest_path} does not exist")
        ckpt = torch.load(latest_path, map_location="cpu")
        if ckpt["model_type"] != cfg.model_type:
            raise ValueError(f"checkpoint is a {ckpt['model_type']} model, config says {cfg.model_type}")

    # ---- data ----
    text, files = load_text(ROOT / cfg.data_dir)
    # When resuming, keep the checkpoint's vocabulary so token ids mean the same thing.
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"]) if ckpt else CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(ids, cfg.val_fraction)

    print(f"files: {[f.name for f in files]}")
    print(f"characters: {len(text):,} | vocab size: {tokenizer.vocab_size}")
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    # ---- model ----
    model = build_model(cfg, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    batch_gen = torch.Generator().manual_seed(cfg.seed)
    start_step, best_val_loss = 0, float("inf")

    if ckpt:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # Restore both random streams (batch sampling and dropout), so a resumed
        # run continues exactly as if it had never stopped.
        batch_gen.set_state(ckpt["batch_rng_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_step, best_val_loss = ckpt["step"], ckpt["best_val_loss"]
        print(f"resumed from {latest_path} at step {start_step}")

    print(f"model: {cfg.model_type} | device: {device} | parameters: {model.num_parameters():,}")
    print(f"expected loss of a uniform guess: ln({tokenizer.vocab_size}) = {math.log(tokenizer.vocab_size):.4f}")

    # ---- output files ----
    tokenizer.save(out_dir / "vocab.json")
    log_file = open(out_dir / "log.csv", "a" if ckpt else "w", newline="")
    log = csv.writer(log_file)
    if not ckpt:
        log.writerow(["step", "train_loss", "val_loss"])
    history = []

    def evaluate(step):
        train_loss = estimate_loss(model, train_data, cfg, device)
        val_loss = estimate_loss(model, val_data, cfg, device)
        log.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}"])
        log_file.flush()
        history.append((step, train_loss, val_loss))
        return train_loss, val_loss

    def save_checkpoint(path, step, train_loss, val_loss):
        torch.save(
            {
                "model_type": cfg.model_type,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "batch_rng_state": batch_gen.get_state(),
                "torch_rng_state": torch.get_rng_state(),
            },
            path,
        )

    # ---- training loop ----
    if not ckpt:
        train_loss, val_loss = evaluate(0)  # loss before any training
        print(f"step {0:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
    start = time.time()

    for step in range(start_step + 1, cfg.max_steps + 1):
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, batch_gen)
        x, y = x.to(device), y.to(device)

        loss = compute_loss(model, x, y)  # forward pass + loss
        loss.backward()  # backward pass: compute gradients
        if cfg.grad_clip > 0:
            # rescale all gradients together if their overall norm exceeds grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()  # update the weights
        optimizer.zero_grad(set_to_none=True)  # reset gradients for the next step

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            train_loss, val_loss = evaluate(step)
            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_path, step, train_loss, val_loss)
                marker = "  * best"
            save_checkpoint(latest_path, step, train_loss, val_loss)
            print(f"step {step:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}{marker}")

    log_file.close()
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s | best val loss {best_val_loss:.4f}")
    print(f"checkpoints: {latest_path}, {best_path}")
    return {"history": history, "best_val_loss": best_val_loss, "elapsed": elapsed,
            "num_parameters": model.num_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to a JSON config")
    parser.add_argument("--model", choices=MODEL_TYPES, help="use configs/<model>_char.json")
    parser.add_argument("--resume", action="store_true", help="continue from <run>/latest.pt")
    parser.add_argument("--max-steps", type=int, help="override max_steps from the config")
    args = parser.parse_args()

    if not args.config and not args.model:
        parser.error("give --model or --config")
    cfg = TrainConfig.from_json(args.config or ROOT / "configs" / f"{args.model}_char.json")
    if args.model and cfg.model_type != args.model:
        parser.error(f"--model {args.model} conflicts with model_type {cfg.model_type!r} in the config")
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
