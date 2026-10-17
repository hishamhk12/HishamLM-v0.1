"""Supervised chat fine-tuning: continue a pretrained checkpoint on (user -> assistant) examples,
with the loss on the assistant's reply only.

Usage (from the project root):
    python -m training.chat_finetune --base checkpoints/RUN/step_20000.pt --out checkpoints/chat_sft

Data: data/chat_history/training/{train,val}.jsonl (built by data.chat_extract), one example per line:
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]}
The last message is the target; earlier messages are context only.

Format (plain text, no new tokens: '<' and '>' are byte-fallback tokens of the existing BPE):
    <USER>\\n{user}\\n<ASSISTANT>\\n{assistant}\\n
Loss mask: only the tokens of the LAST assistant message. No loss on user text, role markers, newlines,
or the earlier turns of a multi-turn example.

Windows: the model sees at most context_length (T) tokens. Every target token is predicted exactly once,
in a window where it has at least T - stride earlier tokens (the same sliding scheme as training.evaluate).
Examples are left-padded with newline tokens (never scored), and generation pads the same way.
Nothing in the model changes: same architecture, tokenizer and checkpoint format (training.evaluate
and inference tools load the result like any other checkpoint).
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from inference.generate import load_checkpoint
from training.train import LN2, ROOT
from training.utils import set_seed

USER, ASSISTANT = "<USER>\n", "<ASSISTANT>\n"
DATA = ROOT / "data/chat_history/training"


def encode_example(tokenizer, messages):
    """Token ids and a loss mask (True = assistant target token) for one example."""
    ids, mask = [], []
    for i, m in enumerate(messages):
        target = i == len(messages) - 1 and m["role"] == "assistant"
        for piece, scored in ((USER if m["role"] == "user" else ASSISTANT, False), (m["content"], target), ("\n", False)):
            t = tokenizer.encode(piece)
            ids += t
            mask += [scored] * len(t)
    return ids, mask


def prompt_ids(tokenizer, messages, context_length):
    """The prompt for the next assistant reply (messages end with a user turn), left-padded like training."""
    ids = []
    for m in messages:
        ids += tokenizer.encode(USER if m["role"] == "user" else ASSISTANT) + tokenizer.encode(m["content"]) + tokenizer.encode("\n")
    ids += tokenizer.encode(ASSISTANT)
    pad = tokenizer.encode("\n")
    return pad * max(context_length - len(ids), 0) + ids


def windows(ids, mask, T, stride, pad_id):
    """(x, y, scored) windows of T tokens: every masked token is scored in exactly one window."""
    ids, mask = [pad_id] * T + ids, [False] * T + mask
    targets = [j for j, m in enumerate(mask) if m]
    out, i = [], 0
    while i < len(targets):
        chunk = [j for j in targets[i : i + stride] if j < targets[i] + stride]
        end = chunk[-1]  # x = ids[end-T : end] predicts y = ids[end-T+1 : end+1]
        scored = set(chunk)
        out.append((ids[end - T : end], ids[end - T + 1 : end + 1], [end - T + 1 + k in scored for k in range(T)]))
        i += len(chunk)
    return out


def load_split(tokenizer, split, T, stride):
    pad_id = tokenizer.encode("\n")[0]
    rows = [json.loads(line) for line in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    ws = [w for r in rows for w in windows(*encode_example(tokenizer, r["messages"]), T, stride, pad_id)]
    x, y, m = (torch.tensor([w[k] for w in ws]) for k in range(3))
    return x, y, m


def masked_nll(model, x, y, m):
    logits = model(x)
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(y.shape)
    return (nll * m).sum(), m.sum()


@torch.no_grad()
def chat_bpc(model, data, char_len, batch_size=64):
    """Bits per character of the assistant targets (the chat held-out loss)."""
    model.eval()
    x, y, m = data
    total, chars = 0.0, 0.0
    for i in range(0, len(x), batch_size):
        nll, _ = masked_nll(model, x[i : i + batch_size], y[i : i + batch_size], m[i : i + batch_size])
        total += nll.item()
        chars += (char_len[y[i : i + batch_size]] * m[i : i + batch_size]).sum().item()
    model.train()
    return total / (LN2 * chars)


def finetune(base, out, lr=3e-5, epochs=3, batch_size=32, warmup=20, eval_interval=20, patience=4, stride=32,
             grad_clip=1.0, seed=42, data=None):
    """data: optional {"train": (x, y, m), "val": (x, y, m)} instead of the jsonl files (tests)."""
    set_seed(seed)
    model, tokenizer, ckpt = load_checkpoint(base, "cpu")
    T = model.context_length
    data = data or {s: load_split(tokenizer, s, T, stride) for s in ("train", "val")}
    char_len = torch.tensor(tokenizer.token_char_lengths(), dtype=torch.float)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = len(data["train"][0])
    steps_per_epoch = math.ceil(n / batch_size)
    max_steps = epochs * steps_per_epoch
    meta = {"base_checkpoint": str(base), "base_step": ckpt["step"], "learning_rate": lr, "epochs": epochs,
            "batch_size": batch_size, "warmup_steps": warmup, "stride": stride, "grad_clip": grad_clip, "seed": seed,
            "train_windows": n, "val_windows": len(data["val"][0]), "steps_per_epoch": steps_per_epoch,
            "train_target_tokens": int(data["train"][2].sum()), "format": "<USER>\\n{user}\\n<ASSISTANT>\\n{assistant}\\n"}

    def save(path, step, val):
        torch.save({"model_type": ckpt["model_type"], "model_state": model.state_dict(), "config": ckpt["config"],
                    "tokenizer": ckpt["tokenizer"], "step": step, "val_bpc": val, "val_metric": "chat_bpc",
                    "chat_finetune": meta}, path)

    model.train()
    best = chat_bpc(model, data["val"], char_len)
    meta["base_chat_val_bpc"] = best
    history = [{"step": 0, "epoch": 0, "train_bpc": float("nan"), "chat_val_bpc": best, "lr": 0.0}]
    print(f"step 0 | chat val {best:.4f} bpc (base)", flush=True)
    save(out / "best.pt", 0, best)
    best_step, bad, step, run_nll, run_chars, start = 0, 0, 0, 0.0, 0.0, time.time()
    stop = False
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            step += 1
            for g in optimizer.param_groups:
                g["lr"] = lr * min(1.0, step / warmup) if warmup else lr
            idx = order[i : i + batch_size]
            x, y, m = (t[idx] for t in data["train"])
            nll, count = masked_nll(model, x, y, m)
            (nll / count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            run_nll += nll.item()
            run_chars += (char_len[y] * m).sum().item()
            if step % eval_interval == 0 or step == max_steps:
                val = chat_bpc(model, data["val"], char_len)
                row = {"step": step, "epoch": round(step / steps_per_epoch, 2), "train_bpc": run_nll / (LN2 * run_chars),
                       "chat_val_bpc": val, "lr": optimizer.param_groups[0]["lr"]}
                history.append(row)
                run_nll, run_chars = 0.0, 0.0
                mark = ""
                if val < best:
                    best, best_step, bad, mark = val, step, 0, "  * best"
                    save(out / "best.pt", step, val)
                else:
                    bad += 1
                print(f"step {step} | epoch {row['epoch']:.2f} | train {row['train_bpc']:.4f} | chat val {val:.4f} bpc{mark}", flush=True)
                if patience and bad >= patience:
                    print(f"early stopping: no chat val improvement for {bad} evaluations (best at step {best_step})")
                    stop = True
                    break
        if stop:
            break
    save(out / "latest.pt", step, history[-1]["chat_val_bpc"])
    meta |= {"steps_done": step, "best_step": best_step, "best_chat_val_bpc": best, "stopped_early": stop,
             "seconds": round(time.time() - start, 1)}
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done | best chat val {best:.4f} bpc at step {best_step} (base {meta['base_chat_val_bpc']:.4f})")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    finetune(args.base, args.out, args.lr, args.epochs, args.batch_size, eval_interval=args.eval_interval,
             patience=args.patience)


if __name__ == "__main__":
    main()
