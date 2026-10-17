"""Build a chat-tuning dataset from a sanitized ChatGPT export (private data; outputs stay out of git).

Usage (from the project root):
    python -m data.chat_extract

1. For every conversation, follow current_node -> parent up to the root: the final, real path. Abandoned
   branches are never read. Only user and assistant text is kept (no system / tool / thoughts /
   reasoning_recap / empty nodes); voice-mode transcriptions count as text.
2. Pairs (user message -> assistant reply) are kept only if they pass every rule below; the reason of every
   exclusion is counted. The assistant reply is never assumed correct:
     - the next user message corrects it ("غلط"، "مو هيك"، "قصدي"، "ما فهمت" ...): the reply is dropped and the
       later, corrected answer is the one that can be kept
     - the reply relied on web search / tool output (citations metadata or raw citation markers)
     - the user message has an image, a file or an attachment, or refers to one
     - code blocks, tables, a mostly non-Arabic reply, an empty / cut-off / very long reply
     - anything sensitive in the user message or the reply: e-mail, phone-like number, URL, password words,
       or a sanitization placeholder ([TOKEN], [PASSWORD], [PATH] ...); the owner's name; personal health
       (symptoms, medicine, tests) in the user's message
   Hallucinations cannot be detected without a reference; the correction rule is the only proxy.
3. Examples: every kept reply is the training target exactly once. If the previous exchange of the same
   conversation was also kept, it is included before it as context (a multi-turn example, no loss on it).
4. Split by whole conversation (90 / 5 / 5); assistant replies that appear in two splits are removed from
   the held-out split.
The user's words are kept as written (dialect, spelling). Assistant replies only lose markdown markup
(**, #, ---) and emoji; nothing is rephrased.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from data.clean import clean_text
from data.prepare import assign_splits

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/chat_history/raw"
OUT = ROOT / "data/chat_history/training"

CORRECTION = re.compile(
    r"غلط|خطأ|خطا\b|مو هيك|مش هيك|ما هيك|مو هيدا|مو هاد|مو صح|مش صح|مو مظبوط|مش مظبوط|مو دقيق|غير صحيح|ليس صحيح|"
    r"قصدي|انا قصدت|مو قصدي|ما قصدت|مو هاد يلي|ليس هذا|ليس ما|ما فهمت|مو فاهم|مش فاهم|"
    r"\bwrong\b|\bincorrect\b|not what i|\bi meant\b|that'?s not|\bno,? that\b", re.IGNORECASE)
REFERS_TO_FILE = re.compile(r"الصورة|بالصورة|هالصورة|الصور المرفقة|السكرين|سكرين شوت|المرفق|الملف المرفق|هالملف|هاد الملف|"
                            r"screenshot|attached|attachment|the image|this image|the file|this file|\bpdf\b", re.IGNORECASE)
CITATION = re.compile(r"[-]|cite|turn\d+(?:search|view|news|file|fetch)|filecite|navlist|【|†")
SENSITIVE = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "url": re.compile(r"https?://|www\."),
    "phone-like number": re.compile(r"(?<!\d)\+?\d[\d\s-]{8,}\d(?!\d)"),
    "password words": re.compile(r"password|passwd|كلمة السر|كلمة المرور|باسورد|باسوورد|رمز التحقق|\botp\b", re.IGNORECASE),
    "sanitization placeholder": re.compile(r"\[[A-Z_]{3,30}\]|<[A-Z_]{3,30}>"),
    "key / token": re.compile(r"\b(?:sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{20,}|Bearer\s+\S+)"),
}
# private details: the account owner's name, and personal health (symptoms, medicine, tests) in the user's message
PERSONAL_NAME = re.compile(r"هشام|تميم|hisham|tamim", re.IGNORECASE)
PERSONAL_HEALTH = re.compile(r"وجع|اوجاع|أوجاع|ألم|الم\b|بيوجعني|عم يوجعني|دكتور|الدكتور|طبيب|دوا\b|دواء|حبوب|تحليل|تحاليل|"
                             r"أعراض|اعراض|حرارة|ضغطي|سكري|التهاب|نزيف|حساسية|مغص|صداع|دوخة|رشح|كحة|سعال|عيادة|مستشفى|"
                             r"عملية جراحية|خدي|صدري|بطني|معدتي|ظهري|راسي", re.IGNORECASE)
ARABIC = re.compile(r"[ء-ي]")
LATIN = re.compile(r"[A-Za-z]")
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")


def text_of(message):
    parts = []
    for p in message["content"].get("parts") or []:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and p.get("content_type") == "audio_transcription":
            parts.append(p.get("text") or "")
    return "\n".join(parts)


def final_path(conv):
    """Messages on the path that ends at current_node, oldest first."""
    mapping, node, path = conv["mapping"], conv.get("current_node"), []
    while node:
        entry = mapping[node]
        if entry.get("message"):
            path.append(entry["message"])
        node = entry.get("parent")
    return path[::-1]


def turns(conv, counts):
    """Consecutive user / assistant text turns (same-role messages merged), with flags."""
    out = []
    for msg in final_path(conv):
        role = msg["author"]["role"]
        ctype = msg["content"].get("content_type")
        if role not in ("user", "assistant") or ctype not in ("text", "multimodal_text"):
            counts[f"skipped message: {role}/{ctype}"] += 1
            continue
        meta = msg.get("metadata") or {}
        text = text_of(msg)
        flags = set()
        if any(isinstance(p, dict) and p.get("content_type") != "audio_transcription" for p in msg["content"].get("parts") or []):
            flags.add("image or file part")
        if meta.get("attachments"):
            flags.add("attachment")
        if meta.get("search_result_groups") or meta.get("content_references") or meta.get("citations"):
            flags.add("web search / citations metadata")
        if not text.strip() and not flags:
            counts["skipped message: empty"] += 1
            continue
        if out and out[-1]["role"] == role:  # e.g. a reply split around a removed tool call
            out[-1]["text"] += "\n\n" + text
            out[-1]["flags"] |= flags | {"merged parts"}
        else:
            out.append({"role": role, "text": text, "flags": flags})
    return out


def normalize_assistant(text):
    text = EMOJI.sub("", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", text)
    return clean_text(text)


def letters_share(text):
    ar, la = len(ARABIC.findall(text)), len(LATIN.findall(text))
    return ar / max(ar + la, 1)


def reply_problem(user, reply, next_user):
    """Why this (user, assistant) pair cannot be a training target, or None."""
    u, a = user["text"], reply["text"]
    if user["flags"] & {"image or file part", "attachment"} or REFERS_TO_FILE.search(u):
        return "user message has / refers to an image or file"
    if "web search / citations metadata" in reply["flags"] or CITATION.search(a):
        return "reply relies on search / tool output (citations)"
    if "merged parts" in reply["flags"]:
        return "reply split around a removed tool call"
    if next_user is not None and CORRECTION.search(next_user["text"][:300]):
        return "user corrected / did not accept the reply"
    if PERSONAL_NAME.search(u) or PERSONAL_NAME.search(a):
        return "private: personal name"
    if PERSONAL_HEALTH.search(u):
        return "private: personal health"
    for name, pattern in SENSITIVE.items():
        if pattern.search(u) or pattern.search(a):
            return f"sensitive: {name}"
    if "```" in a or "`" in a or reply.get("code"):
        return "code in the reply"
    if len(re.findall(r"(?m)^\s*\|.*\|\s*$", a)) >= 2:
        return "table in the reply"
    if letters_share(a) < 0.6:
        return "reply not mostly Arabic"
    if not u.strip():
        return "empty user message"
    if len(u) > 1000:
        return "user message longer than 1000 characters (pasted content)"
    if len(a) < 20:
        return "reply too short"
    if len(a) > 3000:
        return "reply longer than 3000 characters"
    if a.rstrip().endswith((":", "،", ",")):
        return "reply cut off / broken formatting"
    return None


def main():
    convs = []
    for path in sorted(RAW.glob("conversations-*.json")):
        convs += [(path.name, c) for c in json.loads(path.read_text(encoding="utf-8"))]
    counts, reasons = Counter(), Counter()
    examples_by_conv = {}
    for source, conv in convs:
        ts = turns(conv, counts)
        kept_prev = None  # the previous kept exchange (for multi-turn context)
        examples = []
        for i, t in enumerate(ts):
            if t["role"] != "assistant" or i == 0 or ts[i - 1]["role"] != "user":
                continue
            user, reply = ts[i - 1], t
            next_user = next((x for x in ts[i + 1:] if x["role"] == "user"), None)
            counts["pairs on final paths"] += 1
            problem = reply_problem(user, reply, next_user)
            if problem:
                reasons[problem] += 1
                kept_prev = None
                continue
            u, a = clean_text(user["text"]), normalize_assistant(reply["text"])
            messages = []
            if kept_prev is not None and kept_prev[2] == i - 2:
                messages += [{"role": "user", "content": kept_prev[0]}, {"role": "assistant", "content": kept_prev[1]}]
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            examples.append({"messages": messages, "type": "multi" if len(messages) > 2 else "single",
                             "conversation": hashlib.sha256(conv["conversation_id"].encode()).hexdigest()[:16],
                             "source_file": source, "turn": i})
            kept_prev = (u, a, i)
        if examples:
            examples_by_conv[examples[0]["conversation"]] = examples

    # split by whole conversation, by characters
    sizes = [(c, sum(len(m["content"]) for e in ex for m in e["messages"])) for c, ex in examples_by_conv.items()]
    assignment = assign_splits(sizes, 0.05, 0.05, "42:chat")
    splits = {s: [e for c, ex in examples_by_conv.items() if assignment[c] == s for e in ex] for s in ("train", "val", "test")}
    train_replies = {" ".join(e["messages"][-1]["content"].split()) for e in splits["train"]}
    removed = 0
    for s in ("val", "test"):
        before = len(splits[s])
        splits[s] = [e for e in splits[s] if " ".join(e["messages"][-1]["content"].split()) not in train_replies]
        removed += before - len(splits[s])

    OUT.mkdir(parents=True, exist_ok=True)
    for s, ex in splits.items():
        with open(OUT / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for e in ex:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats = {
        "conversations": len(convs), "conversations_with_examples": len(examples_by_conv),
        "message_counts": dict(counts), "excluded_pairs": dict(reasons.most_common()),
        "held_out_replies_also_in_train_removed": removed,
        "splits": {s: {"conversations": sum(assignment[c] == s for c in examples_by_conv), "examples": len(ex),
                       "multi_turn": sum(e["type"] == "multi" for e in ex),
                       "user_chars": sum(len(m["content"]) for e in ex for m in e["messages"] if m["role"] == "user"),
                       "assistant_target_chars": sum(len(e["messages"][-1]["content"]) for e in ex)}
                   for s, ex in splits.items()},
    }
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
