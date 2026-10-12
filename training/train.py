"""Train a language model (MLP or Transformer) from random initialization.

Usage (from the project root):
    python -m training.train --model transformer          # uses configs/transformer_char.json
    python -m training.train --config configs/transformer_corpus.json
    python -m training.train --config configs/phase4/char.json --resume

The main metric is bits per character (BPC): the loss in bits divided by the number of
original characters the predicted tokens stand for. Unlike loss per token, BPC can be
compared between tokenizers (characters vs BPE). For the character tokenizer BPC = loss / ln 2.
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
from tokenizer import tokenizer_from_dict
from training.config import TrainConfig
from training.data import get_batch, legacy_data, prepared_data
from training.utils import get_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
LN2 = math.log(2)


def compute_loss(model, x, y):
    logits = model(x)  # (B, T, V)
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), y.view(B * T))


@torch.no_grad()
def estimate(model, sample_batch, char_len, cfg, device):
    """Loss per token and bits per character over cfg.eval_batches batches from `sample_batch(generator)`.

    A fixed generator means every evaluation sees the same batches,
    so values at different steps are directly comparable.
    """
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    model.eval()
    nll, tokens, chars = 0.0, 0, 0.0
    for _ in range(cfg.eval_batches):
        x, y = sample_batch(gen)
        logits = model(x.to(device))
        nll += F.cross_entropy(logits.view(-1, logits.size(-1)), y.to(device).view(-1), reduction="sum").item()
        tokens += y.numel()
        chars += char_len[y].sum().item()  # characters represented by the predicted tokens
    model.train()
    return nll / tokens, nll / (LN2 * chars)


def learning_rate(step, cfg):
    """Linear warmup, then constant or cosine decay down to min_lr.

    The cosine runs until lr_decay_steps (default: max_steps) and stays at min_lr after it,
    so a run can be extended without the learning rate jumping back up.
    """
    if cfg.warmup_steps and step <= cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    if cfg.lr_schedule == "cosine":
        decay_steps = cfg.lr_decay_steps or cfg.max_steps
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, decay_steps - cfg.warmup_steps))
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
        data = prepared_data(ROOT / cfg.dataset, cfg.sampling,
                             ROOT / cfg.tokenizer if cfg.tokenizer else None, cfg.domain_share)
        if tokenizer is not None and tokenizer.to_dict() != data.tokenizer.to_dict():
            raise ValueError("the checkpoint's tokenizer differs from the configured one")
        return data
    return legacy_data(ROOT / cfg.data_dir, cfg.val_fraction, tokenizer)


def _migrate_log(path, columns, cfg):
    """Rewrite a log from before BPC columns existed (character tokenizer only: BPC = loss / ln 2)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    path.rename(path.with_name("log_before_bpc_columns.csv"))
    with open(path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(columns)
        for r in rows:
            step = int(r["step"])
            new = {"step": step, "learning_rate": r["learning_rate"], "elapsed_time": r["elapsed_time"],
                   "train_loss": r["train_loss"], "val_loss": r["val_loss"],
                   "tokens_seen": step * cfg.batch_size * cfg.context_length}
            new["chars_seen"] = new["tokens_seen"]
            for c in columns:
                if c.endswith("_bpc"):
                    source = c[: -len("_bpc")] if c not in ("train_bpc", "val_bpc") else c.replace("_bpc", "_loss")
                    new[c] = f"{float(r[source]) / LN2:.6g}"
            log.writerow([new.get(c, "") for c in columns])


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
    data = load_data(cfg, tokenizer_from_dict(ckpt["tokenizer"]) if ckpt else None)
    tokenizer = data.tokenizer
    char_len = data.char_len
    print(f"tokenizer: {type(tokenizer).__name__} | vocab size: {tokenizer.vocab_size}")
    print("train tokens: " + ", ".join(f"{n} {len(t):,} (weight {w:.3f})" for n, t, w in
                                        zip(data.names, data.train.values(), data.weights.tolist())))
    print("eval tokens:  " + ", ".join(f"{n} {len(t):,}" for n, t in data.evals.items()))

    # ---- model ----
    model = build_model(cfg, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    batch_gen = torch.Generator().manual_seed(cfg.seed)
    start_step, best_val, best_step, bad_evals, elapsed_before = 0, float("inf"), 0, 0, 0.0
    tokens_seen, chars_seen = 0, 0.0

    if ckpt:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # Restore both random streams (batch sampling and dropout), so a resumed
        # run continues exactly as if it had never stopped.
        batch_gen.set_state(ckpt["batch_rng_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_step = ckpt["step"]
        if ckpt.get("val_metric") == "bpc":
            best_val = ckpt["best_val_bpc"]
        else:  # checkpoints from before BPC: character tokenizer, val loss in nats
            best_val = ckpt["best_val_loss"] / LN2
        best_step, bad_evals = ckpt.get("best_step", 0), ckpt.get("evals_without_improvement", 0)
        elapsed_before = ckpt.get("elapsed_seconds", 0.0)
        tokens_seen = ckpt.get("tokens_seen", start_step * cfg.batch_size * cfg.context_length)
        chars_seen = ckpt.get("chars_seen", float(tokens_seen))  # older runs: character tokenizer
        print(f"resumed from {latest_path} at step {start_step}")

    print(f"model: {cfg.model_type} | device: {device} | parameters: {model.num_parameters():,}")
    print(f"uniform baseline: {math.log(tokenizer.vocab_size):.4f} nats/token")

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
        "tokenizer": type(tokenizer).__name__,
        "tokenizer_file": cfg.tokenizer or None,
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
    columns = (["step", "train_loss", "val_loss", "train_bpc", "val_bpc"]
               + ([f"{n}_bpc" for n in val_names] if len(val_names) > 1 else [])
               + ["learning_rate", "tokens_seen", "chars_seen", "elapsed_time"])
    log_path = out_dir / "log.csv"
    if ckpt and log_path.exists():
        with open(log_path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != columns:
            _migrate_log(log_path, columns, cfg)
    log_file = open(log_path, "a" if ckpt else "w", newline="")
    log = csv.writer(log_file)
    if not ckpt:
        log.writerow(columns)
    history = []
    start = time.time()

    def evaluate(step, lr):
        row = {"step": step}
        row["train_loss"], row["train_bpc"] = estimate(
            model, lambda g: data.train_batch(cfg.batch_size, cfg.context_length, g), char_len, cfg, device)
        losses = {}
        for name in val_names:
            stream = data.evals[name]
            losses[name], row[f"{name}_bpc"] = estimate(model, lambda g, s=stream: _window_batch(s, cfg, g), char_len, cfg, device)
        # validation mixed like the training data: checkpoints are chosen on val_bpc
        row["val_loss"] = sum(w * losses[n] for n, w in data.val_weights.items())
        row["val_bpc"] = sum(w * row[f"{n}_bpc"] for n, w in data.val_weights.items())
        row.update(learning_rate=lr, tokens_seen=tokens_seen, chars_seen=int(chars_seen),
                   elapsed_time=elapsed_before + time.time() - start)
        log.writerow([f"{row[c]:.6g}" if isinstance(row[c], float) else row[c] for c in columns])
        log_file.flush()
        history.append(row)
        return row

    def describe(row):
        parts = [f"step {row['step']:5d}", f"train {row['train_bpc']:.4f}", f"val {row['val_bpc']:.4f} bpc"]
        if len(val_names) > 1:
            parts += [f"{n} {row[n + '_bpc']:.4f}" for n in val_names]
        parts += [f"lr {row['learning_rate']:.2e}", f"{row['chars_seen'] / 1e6:.1f}M chars", f"{row['elapsed_time']:.0f}s"]
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
                "val_bpc": row["val_bpc"],
                "val_metric": "bpc",
                "best_val_bpc": best_val,
                "best_step": best_step,
                "evals_without_improvement": bad_evals,
                "tokens_seen": tokens_seen,
                "chars_seen": chars_seen,
                "elapsed_seconds": row["elapsed_time"],
                "batch_rng_state": batch_gen.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "run_meta": run_meta,
            },
            path,
        )

    # ---- training loop ----
    if not ckpt:
        print(describe(evaluate(0, learning_rate(0, cfg))))  # before any training

    stopped_early = False
    for step in range(start_step + 1, cfg.max_steps + 1):
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data.train_batch(cfg.batch_size, cfg.context_length, batch_gen)
        tokens_seen += x.numel()
        chars_seen += char_len[x].sum().item()  # original characters these tokens stand for
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
            if row["val_bpc"] < best_val:
                best_val, best_step, bad_evals = row["val_bpc"], step, 0
                save_checkpoint(best_path, row)
                marker = "  * best"
            else:
                bad_evals += 1
            save_checkpoint(latest_path, row)
            print(describe(row) + marker, flush=True)
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
        tokens_seen=tokens_seen,
        chars_seen=int(chars_seen),
        best_step=best_step,
        best_val_bpc=best_val,
        stopped_early=stopped_early,
        training_seconds=round(elapsed, 1),
        peak_rss_mb=round(peak_rss_mb, 1),
    )
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {elapsed:.1f}s | best val {best_val:.4f} bpc at step {best_step} | peak RAM {peak_rss_mb:.0f} MB")
    print(f"checkpoints: {latest_path}, {best_path}")
    return {"history": history, "best_val_bpc": best_val, "best_step": best_step, "elapsed": elapsed,
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
