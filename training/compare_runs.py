"""Compare finished runs side by side (tokenizers, sampling mixes, ...).

Usage (from the project root):
    python -m training.compare_runs transformer_corpus transformer_bpe1024 transformer_bpe2048

Reads, for each run in checkpoints/: run_meta.json, log.csv, eval_best.json and samples_best.json
(written by training.train, training.evaluate and inference.sample_eval).
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from model.build import build_model
from training.config import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


def val_bpc_at(log_rows, value, key="chars_seen", scale=1.0):
    """Validation BPC when log column `key` (times `scale`) reached `value` (linear interpolation)."""
    points = [(float(r[key]) * scale, float(r["val_bpc"])) for r in log_rows if r.get(key)]
    for (c0, v0), (c1, v1) in zip(points, points[1:]):
        if c0 <= value <= c1:
            return v0 + (v1 - v0) * (value - c0) / (c1 - c0)
    return None


def flops_per_token(cfg, vocab_size, breakdown):
    """Approximate training FLOPs per token: 6 x (matmul parameters) + attention scores.

    Matmul parameters are the Transformer blocks plus the V x C output projection
    (a tied head still costs the matmul). The embedding lookup is not a matmul.
    """
    matmul = breakdown["attention"] + breakdown["ffn"] + vocab_size * cfg.embedding_dim
    return 6 * matmul + 12 * cfg.n_layers * cfg.context_length * cfg.embedding_dim


def load_run(name, checkpoint_dir="checkpoints"):
    run_dir = ROOT / checkpoint_dir / name
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    with open(run_dir / "log.csv", newline="") as f:
        log = list(csv.DictReader(f))
    ev = json.loads((run_dir / "eval_best.json").read_text(encoding="utf-8"))
    samples_path = run_dir / "samples_best.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8")) if samples_path.exists() else None
    return {"name": name, "meta": meta, "log": log, "eval": ev, "samples": samples}


def summarize(run, train_chars):
    m, r, log = run["meta"], run["eval"]["results"], run["log"]
    cfg = TrainConfig.from_dict(m["config"])
    breakdown = build_model(cfg, m["vocab_size"]).parameter_breakdown() if cfg.model_type == "transformer" else None
    run["flops_per_token"] = flops_per_token(cfg, m["vocab_size"], breakdown) if breakdown else None
    best = next(row for row in log if int(row["step"]) == m["best_step"])
    val = r["A_general_val"]
    s = {
        "run": run["name"],
        "tokenizer": f"{m['tokenizer'].replace('Tokenizer', '')} {m['vocab_size']}",
        "architecture (C x L x H)": f"{cfg.embedding_dim} x {cfg.n_layers} x {cfg.n_heads}",
        "context length (tokens)": cfg.context_length,
        "weight tying": cfg.tie_weights,
        "learning rate": cfg.learning_rate,
        "parameters": m["num_parameters"],
    }
    if breakdown:
        s["params: token emb / pos emb"] = f"{breakdown['token_embedding']:,} / {breakdown['position_embedding']:,}"
        s["params: attention / FFN / LN"] = f"{breakdown['attention']:,} / {breakdown['ffn']:,} / {breakdown['layer_norms']:,}"
        s["params: LM head"] = f"{breakdown['lm_head']:,}" + (" (tied)" if cfg.tie_weights else "")
    s.update({
        "steps (best / done)": f"{m['best_step']} / {m['steps_done']}",
        "tokens processed": m["tokens_seen"],
        "characters represented": m["chars_seen"],
        "characters at best step": int(float(best["chars_seen"])),
        "passes over train text": round(m["chars_seen"] / train_chars, 1),
        "chars per token (A val)": round(val["chars"] / val["tokens"], 2),
        "context in characters": round(m["config"]["context_length"] * val["chars"] / val["tokens"]),
        "training minutes": round(m["training_seconds"] / 60, 1),
        "steps/sec (incl. evals)": round(m["steps_done"] / m["training_seconds"], 2),
        "peak RAM MB": m.get("peak_rss_mb"),
        "est. training PFLOPs": round(run["flops_per_token"] * m["tokens_seen"] / 1e15, 2) if run["flops_per_token"] else None,
        "stopped early": m["stopped_early"],
    })
    for name in ("A_general", "B_domain", "A_ar", "A_en", "A_books"):
        for split in ("val", "test"):
            if f"{name}_{split}" in r:
                s[f"{name}_{split} BPC"] = round(r[f"{name}_{split}"]["bpc"], 4)
    if "A_books_test" in r:
        for split in ("val", "test"):
            # all Arabic text: old Arabic sets + books, weighted by their characters
            parts = [r[f"A_ar_{split}"], r[f"A_books_{split}"]]
            s[f"all Arabic {split} BPC"] = round(sum(p["bpc"] * p["chars"] for p in parts) / sum(p["chars"] for p in parts), 4)
    s["train BPC (sampling mix)"] = round(r["train (sampling mix)"]["bpc"], 4)
    s["gap: A val - train"] = round(r["A_general_val"]["bpc"] - r["train (sampling mix)"]["bpc"], 4)
    if run["samples"]:
        res = run["samples"]["results"]
        s["memorization: max longest match"] = max(x["longest_match"] for x in res)
        s["memorization: median longest match"] = statistics.median(x["longest_match"] for x in res)
        s["memorization: mean 32-gram overlap"] = round(statistics.mean(x["overlap_32gram"] for x in res), 3)
        s["copying warnings"] = sum(x["copying_warning"] for x in res)
        ref = run["samples"]["reference"]
        s["reference (held-out) median / max match"] = f"{ref['longest_match_median']} / {ref['longest_match_max']}"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="checkpoints/phase4_comparison.md")
    parser.add_argument("--dataset", default="data/processed/hishamlm_v1")
    args = parser.parse_args()

    train_chars = json.loads((ROOT / args.dataset / "meta.json").read_text(encoding="utf-8"))["totals"]["train_chars"]
    runs = [load_run(n) for n in args.runs]
    rows = [summarize(r, train_chars) for r in runs]

    # validation BPC at the same number of characters seen, for budgets every run reached
    reached = min(int(r["meta"]["chars_seen"]) for r in runs)
    budgets = [b for b in (5e6, 10e6, 20e6, 40e6, 80e6, 115e6) if b <= reached]
    for row, run in zip(rows, runs):
        for b in budgets:
            v = val_bpc_at(run["log"], b)
            row[f"val BPC at {b / 1e6:.0f}M chars seen"] = round(v, 4) if v is not None else None

    # angle A, equal compute: validation BPC at the same estimated FLOPs and the same wall-clock time
    if all(r["flops_per_token"] for r in runs):
        flops_reached = min(r["flops_per_token"] * int(r["meta"]["tokens_seen"]) for r in runs)
        for row, run in zip(rows, runs):
            for frac in (0.5, 1.0):
                v = val_bpc_at(run["log"], frac * flops_reached, "tokens_seen", run["flops_per_token"])
                row[f"val BPC at {frac * flops_reached / 1e15:.2f} PFLOPs"] = round(v, 4) if v is not None else None
    minutes_reached = min(float(r["log"][-1]["elapsed_time"]) for r in runs) / 60
    for row, run in zip(rows, runs):
        v = val_bpc_at(run["log"], minutes_reached * 60, "elapsed_time")
        row[f"val BPC at {minutes_reached:.0f} min wall-clock"] = round(v, 4) if v is not None else None

    keys = list(dict.fromkeys(k for row in rows for k in row))
    lines = ["| | " + " | ".join(r["run"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for k in keys[1:]:
        lines.append(f"| {k} | " + " | ".join(str(r.get(k)) for r in rows) + " |")
    table = "\n".join(lines)
    (ROOT / args.out).write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
