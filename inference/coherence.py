"""Long-prompt generations and simple, automatic coherence proxies.

Usage (from the project root):
    python -m inference.coherence --checkpoints ctx64=checkpoints/A/best.pt ctx128=checkpoints/B/best.pt \\
                                  --out checkpoints/B/coherence.json

Every checkpoint continues the same prompts with the same seeds (configs/eval_prompts_coherence.json).
Each prompt states a topic or a person at its start, and is longer than 64 tokens, so a model with a
short context no longer sees the beginning of the prompt when it writes.

These are surface measures (word counts and overlaps). They do NOT measure understanding:
  invented_word_rate     words never seen in the training text
  words_per_sentence     average sentence length
  distinct_2             distinct word pairs / all word pairs (low = repetitive)
  repetition_4gram       share of word 4-grams that repeat an earlier one in the same text
  template_repeat        share of sentences starting with the same two words as an earlier sentence
  connectives_per_100w   connecting words (because, however, ثم، لكن ...) per 100 words
  topic_term_rate        the prompt's topic words (listed in the prompt file) per 100 words
  topic_in_last_third    1 if a topic word still appears in the last third of the continuation
  prompt_overlap_first / _last   share of content words found in the prompt, first and second half
  topic_drift            prompt_overlap_first - prompt_overlap_last (higher = drifts away)
  sentence_continuity    content-word overlap (Jaccard) between neighbouring sentences
  abrupt_shift_rate      share of sentences sharing no content word with the previous sentence or the prompt
Memorization (against the training text): longest verbatim match and the source/file it comes
from, share of 32-character and 50-character windows found verbatim in train.
The same text measures on real held-out text of the same length are the reference.
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

import torch

from data.prepare import read_epub
from data.sources import get_source
from inference.generate import load_checkpoint
from inference.memorization import longest_match, ngram_overlap
from inference.sample_eval import generate_chars
from training.train import ROOT

_MARKS = re.compile(r"[ً-ْٰـ]")  # Arabic diacritics, superscript alef, tatweel
_WORD = re.compile(r"[^\W\d_]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟؛])\s+|\n+")
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
STOPWORDS = set("""
في من على إلى عن مع أن إن كان كانت هذا هذه ذلك التي الذي الذين ما لا لم لن قد هو هي هم كل بين بعد قبل عند
أو ثم حتى إذا كما غير أي بل لكن منذ نحو ولا وقد وكان وهو وهي فقد فإن أنه إنه أنها ليس يكون تكون كانوا
the a an and or of to in on at for with by from as is are was were be been it its this that these those
he she they his her their them we you i not no but so if then than which who what when there have has had
will would can could into out up about all one more some other such only also very
""".split())
CONNECTIVES = set("""
لكن ثم لذلك لذا كما إذ حيث بينما أيضا لأن بسبب إلا بالتالي عندما حين إذا فإن أما كذلك رغم لهذا ولهذا بعدما
however therefore because then also but so while although though when thus since after before until meanwhile
""".split())


def normalize(text):
    return _MARKS.sub("", text).lower()


def words(text):
    return _WORD.findall(normalize(text))


def stem(word):
    for p in _PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 3:
            return word[len(p):]
    return word


def content_words(text):
    return {stem(w) for w in words(text) if len(w) >= 3 and w not in STOPWORDS}


def sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if words(s)]


def text_measures(prompt, continuation, lexicon, topic_terms=None):
    ws = words(continuation)
    sents = sentences(continuation)
    bigrams = list(zip(ws, ws[1:]))
    fourgrams = list(zip(ws, ws[1:], ws[2:], ws[3:]))
    seen, repeats = set(), 0
    for g in fourgrams:
        repeats += g in seen
        seen.add(g)
    starts, template = set(), 0
    for s in sents:
        head = tuple(words(s)[:2])
        if len(head) == 2:
            template += head in starts
            starts.add(head)
    prompt_words = content_words(prompt)
    half = len(continuation) // 2

    def overlap(text):
        cw = [stem(w) for w in words(text) if len(w) >= 3 and w not in STOPWORDS]
        return sum(w in prompt_words for w in cw) / max(len(cw), 1)

    sent_words = [content_words(s) for s in sents]
    jaccard = [len(a & b) / len(a | b) for a, b in zip(sent_words, sent_words[1:]) if a | b]
    later = [(prev, cur) for prev, cur in zip(sent_words, sent_words[1:]) if len(cur) >= 3]
    abrupt = [not (cur & prev) and not (cur & prompt_words) for prev, cur in later]
    connective = sum(w in CONNECTIVES or (w[:1] in "وف" and w[1:] in CONNECTIVES) for w in ws)
    result = {
        "words": len(ws),
        "invented_word_rate": sum(w not in lexicon for w in ws) / max(len(ws), 1),
        "words_per_sentence": len(ws) / max(len(sents), 1),
        "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
        "repetition_4gram": repeats / max(len(fourgrams), 1),
        "template_repeat": template / max(len(sents), 1),
        "connectives_per_100w": 100 * connective / max(len(ws), 1),
        "prompt_overlap_first": overlap(continuation[:half]),
        "prompt_overlap_last": overlap(continuation[half:]),
        "sentence_continuity": statistics.mean(jaccard) if jaccard else 0.0,
        "abrupt_shift_rate": sum(abrupt) / len(abrupt) if abrupt else 0.0,
    }
    result["topic_drift"] = result["prompt_overlap_first"] - result["prompt_overlap_last"]
    if topic_terms:
        terms = [normalize(t) for t in topic_terms]
        text = normalize(continuation)
        result["topic_term_rate"] = 100 * sum(text.count(t) for t in terms) / max(len(ws), 1)
        result["topic_in_last_third"] = float(any(t in text[2 * len(text) // 3:] for t in terms))
    return result


class Attribution:
    """Which training source, and which file inside it, a verbatim match comes from."""

    def __init__(self, dataset):
        self.sources = {p.stem: p.read_text(encoding="utf-8") for p in sorted((dataset / "train").glob("*.txt"))}
        self.files = {"after_sales": {p.name: p.read_text(encoding="utf-8")
                                      for p in sorted((ROOT / "data/processed/domain_after_sales/docs").glob("*.txt"))}}
        books_dir = ROOT / "data/raw/private_books/hindawi_books_ar"
        if books_dir.exists():
            books = get_source("hindawi_books_ar")["files"]
            self.files["hindawi_books_ar"] = {b["file"]: read_epub(books_dir / b["file"], b["title"])[0] for b in books}

    def __call__(self, match):
        if len(match) < 20:
            return None
        for name, text in self.sources.items():
            if match in text:
                inside = [f for f, t in self.files.get(name, {}).items() if match in t]
                return {"source": name, "file": inside[0] if inside else None}
        return None


def summarize(rows, keys):
    return {k: round(statistics.mean(r[k] for r in rows), 4) for k in keys if all(k in r for r in rows)} | {"samples": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="label=path/to/checkpoint.pt")
    parser.add_argument("--prompts", default="configs/eval_prompts_coherence.json")
    parser.add_argument("--dataset", default="data/processed/hishamlm_v2")
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--max-new-chars", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    attribution = Attribution(dataset)
    train_text = "\n".join(attribution.sources.values())
    lexicon = set(words(train_text))
    prompts = json.loads((ROOT / args.prompts).read_text(encoding="utf-8"))
    for items in prompts.values():
        for item in items:
            length, _ = longest_match(item["prompt"], train_text)
            if length >= 60:
                raise ValueError(f"prompt shares {length} characters with the training text: {item['prompt'][:60]}")

    text_keys = ["invented_word_rate", "words_per_sentence", "distinct_2", "repetition_4gram", "template_repeat",
                 "connectives_per_100w", "topic_term_rate", "topic_in_last_third", "prompt_overlap_first",
                 "prompt_overlap_last", "topic_drift", "sentence_continuity", "abrupt_shift_rate"]
    mem_keys = ["longest_match", "overlap_32gram", "overlap_50char"]
    report = {"settings": vars(args), "models": {}, "reference": {}}

    for spec in args.checkpoints:
        label, path = spec.split("=", 1)
        model, tokenizer, ckpt = load_checkpoint(path, "cpu")
        rows = []
        for group, items in prompts.items():
            for p_idx, item in enumerate(items):
                for r in range(args.samples_per_prompt):
                    seed = args.seed + len(rows)  # the same seeds for every checkpoint
                    gen = torch.Generator().manual_seed(seed)
                    cont = generate_chars(model, tokenizer, item["prompt"], args.max_new_chars,
                                          args.temperature, args.top_k, gen)
                    length, match = longest_match(cont, train_text)
                    rows.append({"group": group, "prompt_index": p_idx, "seed": seed, "continuation": cont,
                                 **text_measures(item["prompt"], cont, lexicon, item.get("topic_terms")),
                                 "longest_match": length, "longest_match_text": match,
                                 "match_source": attribution(match),
                                 "overlap_32gram": ngram_overlap(cont, train_text, 32, 4),
                                 "overlap_50char": ngram_overlap(cont, train_text, 50, 5)})
            print(f"{label}: {group} done", flush=True)
        groups = {g: summarize([r for r in rows if r["group"] == g], text_keys + mem_keys) for g in prompts}
        for g in groups:
            groups[g]["longest_match_max"] = max(r["longest_match"] for r in rows if r["group"] == g)
        report["models"][label] = {"checkpoint": path, "step": ckpt["step"],
                                   "context_length": ckpt["config"]["context_length"],
                                   "groups": groups, "all": summarize(rows, text_keys + mem_keys), "samples": rows}

    # reference: real held-out text, a 300-character "prompt" followed by the next max_new_chars characters
    source_groups = {"arabic": ["common_voice_ar", "ntrex_ar", "xquad_ar"], "english": ["common_voice_en", "gutenberg_en"],
                     "after_sales": ["after_sales"], "books": ["hindawi_books_ar"]}
    rng = random.Random(args.seed)
    for group, names in source_groups.items():
        held = "\n".join((dataset / s / f"{n}.txt").read_text(encoding="utf-8") for s in ("val", "test") for n in names)
        rows = []
        for _ in range(15):
            start = rng.randrange(300, len(held) - args.max_new_chars)
            cont = held[start : start + args.max_new_chars]
            length, match = longest_match(cont, train_text)
            rows.append({**text_measures(held[start - 300 : start], cont, lexicon), "longest_match": length,
                         "overlap_32gram": ngram_overlap(cont, train_text, 32, 4),
                         "overlap_50char": ngram_overlap(cont, train_text, 50, 5)})
        report["reference"][group] = summarize(rows, text_keys + mem_keys)

    show = ["invented_word_rate", "words_per_sentence", "distinct_2", "repetition_4gram", "template_repeat",
            "connectives_per_100w", "topic_term_rate", "topic_in_last_third", "topic_drift",
            "sentence_continuity", "abrupt_shift_rate", "longest_match", "overlap_32gram", "overlap_50char"]
    for group in prompts:
        print(f"\n== {group}")
        print(f"{'':12s}" + "".join(f"{k[:11]:>12s}" for k in show))
        named = [(label, m["groups"][group]) for label, m in report["models"].items()]
        for label, r in named + [("reference", report["reference"][group])]:
            print(f"{label:12s}" + "".join(f"{r[k]:12.3f}" if k in r else f"{'-':>12s}" for k in show))
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
