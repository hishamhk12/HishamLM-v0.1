"""Train a language model (MLP or Transformer) from random initialization.

Usage (from the project root):
    python -m training.train --model transformer          # uses configs/transformer_char.json
    python -m training.train --config configs/transformer_corpus.json
    python -m training.train --config configs/transformer_corpus.json --resume --max-steps 8000
"""

import argparse
import csv
import json
import math
import os
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from model.build import MODEL_TYPES, build_model
from tokenizer.char_tokenizer import CharTokenizer
from training.config import TrainConfig
from training.data import get_batch, legacy_data, prepared_data
from training.utils import get_device, set_seed

ROOT = Path(__file__).resolve().parents[1]


def compute_loss(model, x, y):
    logits = model(x)  # (B, T, V)
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), y.view(B * T))


@torch.no_grad()
def estimate_loss(model, sample_batch, cfg, device):
    """Average loss over cfg.eval_batches batches from `sample_batch(generator)`.

    A fixed generator means every evaluation sees the same batches,
    so losses at different steps are directly comparable.
    """
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = sample_batch(gen)
        losses.append(compute_loss(model, x.to(device), y.to(device)).item())
    model.train()
    return sum(losses) / len(losses)


def learning_rate(step, cfg):
    """Linear warmup, then constant or cosine decay down to min_lr."""
    if cfg.warmup_steps and step <= cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    if cfg.lr_schedule == "cosine":
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress))
    return cfg.learning_rate


def git_state():
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def load_data(cfg, tokenizer=None):
    if cfg.dataset:
        data = prepared_data(ROOT / cfg.dataset, cfg.sampling)
        if tokenizer is not None and tokenizer.to_dict() != data.tokenizer.to_dict():
            raise ValueError("the checkpoint's vocabulary differs from the dataset's vocab.json")
        return data
    return legacy_data(ROOT / cfg.data_dir, cfg.val_fraction, tokenizer)


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
    # When resuming, keep the checkpoint's vocabulary so token ids mean the same thing.
    data = load_data(cfg, CharTokenizer.from_dict(ckpt["tokenizer"]) if ckpt else None)
    tokenizer = data.tokenizer
    print(f"vocab size: {tokenizer.vocab_size}")
    print("train tokens: " + ", ".join(f"{n} {len(t):,} (weight {w:.3f})" for n, t, w in
                                        zip(data.names, data.train.values(), data.weights.tolist())))
    print("eval tokens:  " + ", ".join(f"{n} {len(t):,}" for n, t in data.evals.items()))

    # ---- model ----
    model = build_model(cfg, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    batch_gen = torch.Generator().manual_seed(cfg.seed)
    start_step, best_val_loss, best_step, bad_evals, elapsed_before = 0, float("inf"), 0, 0, 0.0

    if ckpt:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # Restore both random streams (batch sampling and dropout), so a resumed
        # run continues exactly as if it had never stopped.
        batch_gen.set_state(ckpt["batch_rng_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_step, best_val_loss = ckpt["step"], ckpt["best_val_loss"]
        best_step, bad_evals = ckpt.get("best_step", 0), ckpt.get("evals_without_improvement", 0)
        elapsed_before = ckpt.get("elapsed_seconds", 0.0)
        print(f"resumed from {latest_path} at step {start_step}")

    uniform = math.log(tokenizer.vocab_size)
    print(f"model: {cfg.model_type} | device: {device} | parameters: {model.num_parameters():,}")
    print(f"uniform baseline loss: ln({tokenizer.vocab_size}) = {uniform:.4f}")
    for name, b in data.info.get("baselines", {}).items():
        if "unigram_baseline_loss" in b:
            print(f"unigram baseline loss on {name}: {b['unigram_baseline_loss']:.4f}")

    # ---- run metadata (for comparing experiments later) ----
    commit, dirty = git_state()
    run_meta = {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "git_commit": commit,
        "git_dirty": dirty,
        "dataset": data.info.get("dataset", cfg.data_dir),
        "dataset_fingerprint": data.info.get("fingerprint"),
        "sampling_weights": dict(zip(data.names, [round(w, 4) for w in data.weights.tolist()])),
        "vocab_size": tokenizer.vocab_size,
        "num_parameters": model.num_parameters(),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resumed_from_step": start_step if ckpt else None,
    }

    # ---- output files ----
    tokenizer.save(out_dir / "vocab.json")
    val_names = list(data.val_weights)
    columns = ["step", "train_loss", "val_loss"] + (val_names if len(val_names) > 1 else []) + ["learning_rate", "elapsed_time"]
    log_file = open(out_dir / "log.csv", "a" if ckpt else "w", newline="")
    log = csv.writer(log_file)
    if not ckpt:
        log.writerow(columns)
    history = []
    start = time.time()

    def evaluate(step, lr):
        losses = {"train": estimate_loss(model, lambda g: data.train_batch(cfg.batch_size, cfg.context_length, g), cfg, device)}
        for name in val_names:
            stream = data.evals[name]
            losses[name] = estimate_loss(model, lambda g, s=stream: _window_batch(s, cfg, g), cfg, device)
        val_loss = sum(w * losses[n] for n, w in data.val_weights.items())
        elapsed = elapsed_before + time.time() - start
        row = {"step": step, "train_loss": losses["train"], "val_loss": val_loss, **losses,
               "learning_rate": lr, "elapsed_time": elapsed}
        log.writerow([f"{row[c]:.6g}" if isinstance(row[c], float) else row[c] for c in columns])
        log_file.flush()
        history.append(row)
        return row

    def describe(row):
        parts = [f"step {row['step']:5d}", f"train {row['train_loss']:.4f}", f"val {row['val_loss']:.4f}"]
        if len(val_names) > 1:
            parts += [f"{n} {row[n]:.4f}" for n in val_names]
        parts += [f"lr {row['learning_rate']:.2e}", f"{row['elapsed_time']:.0f}s"]
        return " | ".join(parts)

    def save_checkpoint(path, row):
        torch.save(
            {
                "model_type": cfg.model_type,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "step": row["step"],
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss"],
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "evals_without_improvement": bad_evals,
                "elapsed_seconds": row["elapsed_time"],
                "batch_rng_state": batch_gen.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "run_meta": run_meta,
            },
            path,
        )

    # ---- training loop ----
    if not ckpt:
        print(describe(evaluate(0, learning_rate(0, cfg))))  # loss before any training

    stopped_early = False
    for step in range(start_step + 1, cfg.max_steps + 1):
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data.train_batch(cfg.batch_size, cfg.context_length, batch_gen)
        x, y = x.to(device), y.to(device)

        loss = compute_loss(model, x, y)  # forward pass + loss
        loss.backward()  # backward pass: compute gradients
        if cfg.grad_clip > 0:
            # rescale all gradients together if their overall norm exceeds grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()  # update the weights
        optimizer.zero_grad(set_to_none=True)  # reset gradients for the next step

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            row = evaluate(step, lr)
            marker = ""
            if row["val_loss"] < best_val_loss:
                best_val_loss, best_step, bad_evals = row["val_loss"], step, 0
                save_checkpoint(best_path, row)
                marker = "  * best"
            else:
                bad_evals += 1
            save_checkpoint(latest_path, row)
            print(describe(row) + marker)
            if cfg.early_stopping_patience and bad_evals >= cfg.early_stopping_patience:
                print(f"early stopping: no val improvement for {bad_evals} evaluations (best at step {best_step})")
                stopped_early = True
                break

    log_file.close()
    elapsed = elapsed_before + time.time() - start
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    run_meta.update(
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        steps_done=history[-1]["step"] if history else start_step,
        best_step=best_step,
        best_val_loss=best_val_loss,
        stopped_early=stopped_early,
        training_seconds=round(elapsed, 1),
        peak_rss_mb=round(peak_rss_mb, 1),
    )
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {elapsed:.1f}s | best val loss {best_val_loss:.4f} at step {best_step} | peak RAM {peak_rss_mb:.0f} MB")
    print(f"checkpoints: {latest_path}, {best_path}")
    return {"history": history, "best_val_loss": best_val_loss, "best_step": best_step, "elapsed": elapsed,
            "num_parameters": model.num_parameters(), "stopped_early": stopped_early}


def _window_batch(stream, cfg, generator):
    return get_batch(stream, cfg.batch_size, cfg.context_length, generator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to a JSON config")
    parser.add_argument("--model", choices=MODEL_TYPES, help="use configs/<model>_char.json")
    parser.add_argument("--resume", action="store_true", help="continue from <run>/latest.pt")
    parser.add_argument("--max-steps", type=int, help="override max_steps from the config")
    parser.add_argument("--run-name", help="override run_name (checkpoint folder) from the config")
    args = parser.parse_args()

    if not args.config and not args.model:
        parser.error("give --model or --config")
    cfg = TrainConfig.from_json(args.config or ROOT / "configs" / f"{args.model}_char.json")
    if args.model and cfg.model_type != args.model:
        parser.error(f"--model {args.model} conflicts with model_type {cfg.model_type!r} in the config")
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.run_name:
        cfg.run_name = args.run_name

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
