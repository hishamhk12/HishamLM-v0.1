"""Before / after evaluation of chat fine-tuning: held-out chat loss, generated replies, simple proxies, memorization.

Usage (from the project root):
    python -m inference.chat_eval --checkpoints base=checkpoints/RUN/step_20000.pt sft=checkpoints/chat_sft/X/best.pt \\
                                  --out checkpoints/chat_sft/chat_eval.json

Every checkpoint answers the same prompts with the same seeds, in the training format (<USER> / <ASSISTANT>).
Follow-ups are real two-turn conversations: the model's own first reply is the context of the second turn.
A reply ends when the model writes a role marker, or at --max-chars characters.

Surface measures (they do NOT measure understanding or correctness; read the samples):
  invented words         Arabic words not in the pretraining lexicon (names from Wikipedia titles accepted)
  fragments, words/sentence, repeated openings, Arabic share
  dialect per 100 words  Levantine words (شو، ليش، هيك، كتير، بدك ...); compared with the real assistant replies
                         and the real user messages: far above the replies = imitating the user's dialect
  echoes the prompt      the reply repeats 15+ characters of the user's message
  on topic               the reply mentions the prompt's topic words (for follow-ups: the first turn's topic)
  instruction proxies    points: 2+ bullet / numbered lines; example: an "example" word; short: the reply length
Memorization against the chat TRAIN split: longest verbatim match with the training replies and with the
training user messages, share of 50-character windows found in the training conversations, replies with a
match of 80+ characters, and leaks (owner's name, e-mail, phone, URL). Real held-out replies are the reference.
"""

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path

import torch

from data.chat_extract import PERSONAL_NAME, SENSITIVE
from inference.coherence import normalize, sentences
from inference.generate import generate, load_checkpoint
from inference.memorization import longest_match, ngram_overlap
from inference.wiki_style_eval import arabic_words
from training.chat_finetune import ASSISTANT, USER, chat_bpc, load_split, prompt_ids
from training.train import ROOT

DATA = ROOT / "data/chat_history/training"
SINGLE = ["مرحبا كيفك؟", "اشرحلي شو يعني API", "ليش الذهب بيتأثر بالفائدة؟", "عندي مشكلة بأودو شو ممكن يكون السبب؟",
          "اشرحلي هالموضوع ببساطة", "ما فهمت، وضحلي أكتر"]
FOLLOW_UPS = [("شو يعني API؟", "طيب اعطيني مثال من أودو"),
              ("ليش الذهب بيتأثر بالفائدة؟", "اختصرلي الجواب"),
              ("ليش الذهب بيتأثر بالفائدة؟", "اعطيني نقاط"),
              ("ليش الذهب بيتأثر بالفائدة؟", "اشرح بدون تعقيد"),
              ("ليش الذهب بيتأثر بالفائدة؟", "اعطيني مثال")]
TOPICS = {"API": ["api", "واجه", "برمج", "تطبيق", "نظام"], "الذهب": ["ذهب", "فائد", "فايد"],
          "أودو": ["أودو", "اودو", "odoo", "نظام", "مشكل"], "مرحبا": ["مرحب", "أهلا", "اهلا", "منيح", "بخير", "الحمد"]}
DIALECT = set("""شو ليش هيك كتير منيح هلق هلأ بدك بدي بدنا عم مو مش اشي شي هاد هاي هيدا هيدي لهيك كمان طيب ازا إزا
فيك فيني تبعك تبع بحكي منشان مشان لسا لسه هون هنيك شلون وين ليه ايش إيش عشان علشان مين كيفك""".split())
BULLET = re.compile(r"(?m)^\s*(?:[-•*▪●]|\d+[.)\-]|[٠-٩]+[.)\-])\s*\S")
EXAMPLE = re.compile(r"مثال|مثلا|مثلًا|مثلاً|سبيل المثال|تخيل|متل")


def plain_ids(tokenizer, messages, context_length):
    """Reference prompt without role markers: the turns one per line (how the base model saw text)."""
    ids = tokenizer.encode("\n".join(m["content"] for m in messages) + "\n")
    return tokenizer.encode("\n") * max(context_length - len(ids), 0) + ids


def reply(model, tokenizer, messages, max_chars, temperature, top_k, seed, plain=False):
    """Generate the assistant reply to `messages`; stops at a role marker."""
    ids = torch.tensor([(plain_ids if plain else prompt_ids)(tokenizer, messages, model.context_length)])
    n, gen = ids.size(1), torch.Generator().manual_seed(seed)
    stopped = False
    while True:
        ids = generate(model, ids, 1, temperature, top_k, gen)
        text = tokenizer.decode(ids[0, n:].tolist())
        cut = min((text.find(m) for m in ("<USER", "<ASSISTANT") if m in text), default=-1)
        if cut >= 0:
            text, stopped = text[:cut], True
            break
        if len(text) >= max_chars:
            text = text[:max_chars]
            break
    return text.strip(), stopped


def topic_hit(text, topic):
    t = normalize(text)
    return float(any(w in t for w in TOPICS[topic])) if topic else None


def measures(text, user, lexicon, names, train):
    ws = arabic_words(text)
    sents = sentences(text)
    heads = [tuple(arabic_words(s)[:2]) for s in sents]
    letters = re.findall(r"[ء-يA-Za-z]", text)
    unknown = [w for w in ws if w not in lexicon]
    row = {
        "chars": len(text),
        "arabic_share": sum("ء" <= c <= "ي" for c in letters) / max(len(letters), 1),
        "invented": sum(w not in names for w in unknown) / max(len(ws), 1),
        "words_per_sentence": len(ws) / max(len(sents), 1),
        "fragments": sum(len(arabic_words(s)) < 4 for s in sents) / max(len(sents), 1),
        "repeated_openings": sum(heads[i] in heads[:i] for i in range(len(heads)) if len(heads[i]) == 2) / max(len(sents), 1),
        "dialect_per_100w": 100 * sum(w in DIALECT for w in ws) / max(len(ws), 1),
        "echoes_prompt": float(user is not None and longest_match(text, user)[0] >= 15),
        "bullets": float(len(BULLET.findall(text)) >= 2),
        "example_word": float(bool(EXAMPLE.search(text))),
    }
    a_len, a_match = longest_match(text, train["assistant"])
    u_len, _ = longest_match(text, train["user"])
    row |= {"match_train_replies": a_len, "match_train_user": u_len,
            "overlap_50char": ngram_overlap(text, train["all"], 50, 5), "copy_80": float(a_len >= 80 or u_len >= 80),
            "leak": float(bool(PERSONAL_NAME.search(text)) or any(p.search(text) for k, p in SENSITIVE.items()
                                                                  if k in ("email", "url", "phone-like number")))}
    if a_len >= 50:
        row["longest_match_text"] = a_match
    return row


def summarize(rows):
    keys = [k for k in rows[0] if k not in ("on_topic", "seed") and all(isinstance(r.get(k), (int, float)) for r in rows)]
    out = {k: round(statistics.mean(r[k] for r in rows), 4) for k in keys}
    out["match_train_replies_max"] = max(r["match_train_replies"] for r in rows)
    out["samples"] = len(rows)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="label=path, or label=path#plain (prompt without role markers)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dataset", default="data/processed/hishamlm_v2_wiki20m")
    parser.add_argument("--titles", default="data/wikipedia_ar/raw/arwiki-latest-all-titles-in-ns0.gz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)

    lexicon = set(arabic_words("\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / args.dataset / "train").glob("*.txt")))))
    with gzip.open(ROOT / args.titles, "rt", encoding="utf-8") as f:
        names = {w for line in f for w in arabic_words(line.replace("_", " "))} - lexicon
    rows = [json.loads(line) for line in open(DATA / "train.jsonl", encoding="utf-8")]
    train = {"assistant": "\n".join(r["messages"][-1]["content"] for r in rows),
             "user": "\n".join(m["content"] for r in rows for m in r["messages"] if m["role"] == "user")}
    train["all"] = train["assistant"] + "\n" + train["user"]

    # reference: real held-out replies, cut like the generations
    held = [json.loads(line) for s in ("val", "test") for line in open(DATA / f"{s}.jsonl", encoding="utf-8")]
    ref = [measures(r["messages"][-1]["content"][: args.max_chars], r["messages"][-2]["content"], lexicon, names, train) for r in held]
    report = {"settings": vars(args), "reference_heldout_replies": summarize(ref),
              "reference_train_user_dialect_per_100w": round(100 * sum(w in DIALECT for w in arabic_words(train["user"]))
                                                             / len(arabic_words(train["user"])), 4),
              "reference_train_reply_dialect_per_100w": round(100 * sum(w in DIALECT for w in arabic_words(train["assistant"]))
                                                              / len(arabic_words(train["assistant"])), 4),
              "models": {}}

    for spec in args.checkpoints:
        label, path = spec.split("=", 1)
        path, _, fmt = path.partition("#")
        plain = fmt == "plain"
        model, tokenizer, _ = load_checkpoint(path, "cpu")
        T = model.context_length
        char_len = torch.tensor(tokenizer.token_char_lengths(), dtype=torch.float)
        res = {"checkpoint": path, "chat_bpc": {s: chat_bpc(model, load_split(tokenizer, s, T, 32), char_len) for s in ("val", "test")},
               "samples": []}
        print(label, res["chat_bpc"], flush=True)
        for seed in range(args.seeds):
            for p in SINGLE:
                topic = next((t for t in TOPICS if t in p), None) or ("مرحبا" if "مرحبا" in p else None)
                text, stopped = reply(model, tokenizer, [{"role": "user", "content": p}], args.max_chars,
                                      args.temperature, args.top_k, seed, plain)
                res["samples"].append({"group": "single", "prompt": p, "seed": seed, "reply": text, "stopped": float(stopped),
                                       "on_topic": topic_hit(text, topic), **measures(text, p, lexicon, names, train)})
            for first, second in FOLLOW_UPS:
                topic = "API" if "API" in first else "الذهب"
                a1, _ = reply(model, tokenizer, [{"role": "user", "content": first}], args.max_chars, args.temperature, args.top_k, seed, plain)
                msgs = [{"role": "user", "content": first}, {"role": "assistant", "content": a1}, {"role": "user", "content": second}]
                text, stopped = reply(model, tokenizer, msgs, args.max_chars, args.temperature, args.top_k, seed, plain)
                visible = tokenizer.decode((plain_ids if plain else prompt_ids)(tokenizer, msgs, T)[-T:])
                res["samples"].append({"group": "follow_up", "prompt": f"{first} → {second}", "seed": seed, "first_reply": a1,
                                       "reply": text, "stopped": float(stopped), "first_turn_visible": float(first in visible),
                                       "on_topic": topic_hit(text, topic), **measures(text, second, lexicon, names, train)})
        res["summary"] = {g: summarize([s for s in res["samples"] if g in ("all", s["group"])]) for g in ("all", "single", "follow_up")}
        for g in ("all", "single", "follow_up"):
            sel = [s for s in res["samples"] if g in ("all", s["group"])]
            res["summary"][g]["stopped"] = round(statistics.mean(s["stopped"] for s in sel), 4)
            res["summary"][g]["on_topic"] = round(statistics.mean(s["on_topic"] for s in sel if s["on_topic"] is not None), 4)
        by_instruction = {}
        for s in res["samples"]:
            if s["group"] == "follow_up":
                k = s["prompt"].split(" → ")[1]
                by_instruction.setdefault(k, []).append(s)
        res["instructions"] = {k: {"bullets": statistics.mean(s["bullets"] for s in v), "example_word": statistics.mean(s["example_word"] for s in v),
                                   "chars": statistics.mean(s["chars"] for s in v), "stopped": statistics.mean(s["stopped"] for s in v),
                                   "first_turn_visible": statistics.mean(s["first_turn_visible"] for s in v)}
                               for k, v in by_instruction.items()}
        report["models"][label] = res
        print(json.dumps(res["summary"], ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
