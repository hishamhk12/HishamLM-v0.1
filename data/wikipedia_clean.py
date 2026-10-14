"""Conservative MediaWiki markup -> plain text, for Arabic Wikipedia articles.

Removed: comments, <ref> references, templates {{...}} (infoboxes, navigation boxes, citations, ...),
tables {| ... |}, file / image links, categories, interlanguage links, math / code / gallery blocks,
HTML tags (their inner text is kept), bold / italic quotes, behaviour switches (__TOC__),
and the sections that only hold links or references (المراجع، وصلات خارجية، انظر أيضًا، ...).
Kept: the article prose, headings (as their own line), and list items that read as sentences.
Links [[target|label]] become "label", external links [url label] become "label".
The text is never rephrased; character changes: HTML entities decoded, presentation forms -> letters.
"""

import html
import re
import unicodedata

from data.clean import clean_text

FILE_NS = ("ملف", "صورة", "file", "image", "وسائط", "media")
CATEGORY_NS = ("تصنيف", "category")
DROP_SECTIONS = {"مراجع", "المراجع", "مصادر", "المصادر", "المراجع والمصادر", "مصادر ومراجع", "وصلات خارجية",
                 "روابط خارجية", "انظر أيضا", "انظر أيضًا", "اقرأ أيضا", "اقرأ أيضًا", "قراءات إضافية", "قراءات أخرى",
                 "هوامش", "الهوامش", "ملاحظات", "حواشي", "الحواشي", "مراجع وملاحظات", "ملاحظات ومراجع", "معرض صور",
                 "معرض الصور", "المراجع والهوامش", "وصلات", "references", "external links", "see also", "notes",
                 "further reading", "bibliography", "sources"}
REMOVE_BLOCKS = ("ref", "gallery", "math", "chem", "ce", "score", "timeline", "imagemap", "mapframe", "maplink",
                 "graph", "syntaxhighlight", "source", "code", "pre", "templatedata", "hiero", "nowiki", "references",
                 "inputbox", "categorytree", "charinsert", "noinclude", "includeonly", "templatestyles")
PRESENTATION_FORM = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
HEADING = "\x00H:"  # marks heading lines while sections are processed
INTERWIKI = re.compile(r"^:?[a-z]{2,3}(-[a-z]+)?:", re.IGNORECASE)


def remove_nested(text, opener, closer, stats=None, key=None, names=None):
    """Remove balanced opener ... closer blocks (they may nest). Counts removed blocks / characters."""
    out, depth, i, start = [], 0, 0, 0
    n_o, n_c = len(opener), len(closer)
    while i < len(text):
        if text.startswith(opener, i):
            if depth == 0:
                out.append(text[start:i])
                block_start = i
            depth += 1
            i += n_o
        elif depth and text.startswith(closer, i):
            depth -= 1
            i += n_c
            if depth == 0:
                if stats is not None:
                    stats[key + "_count"] += 1
                    stats[key + "_chars"] += i - block_start
                    if names is not None:
                        name = text[block_start + n_o:block_start + n_o + 60].split("|")[0].split("\n")[0].strip()
                        names[name] += 1
                start = i
        else:
            i += 1
    if depth:  # unbalanced: drop the rest only if it is short, otherwise keep it (and it shows as broken markup)
        if stats is not None:
            stats["unbalanced_" + key] += 1
        rest = text[block_start:]
        out.append("" if len(rest) < 2000 else rest)
    else:
        out.append(text[start:])
    return "".join(out)


def replace_links(text, categories):
    """Innermost [[...]] first, so captions with links inside file links are handled."""
    pattern = re.compile(r"\[\[([^\[\]]*)\]\]")

    def repl(m):
        inner = m.group(1)
        target, _, label = inner.partition("|")
        t = target.strip()
        low = t.lower().lstrip(":")
        ns = low.split(":", 1)[0].strip() if ":" in low else ""
        if ns in FILE_NS:
            return ""
        if ns in CATEGORY_NS:
            if not t.startswith(":"):
                categories.append(t.split(":", 1)[1].split("|")[0].strip())
                return ""
            return label or t.split(":", 1)[1]
        if INTERWIKI.match(t) and not label:
            return ""  # interlanguage link
        if label:
            return label.split("|")[-1]
        return t.lstrip(":").split("#")[0] or t.split("#")[-1]

    for _ in range(10):
        new = pattern.sub(repl, text)
        if new == text:
            break
        text = new
    return text


# Inline templates that print text inside a sentence: keep that text instead of deleting it.
# rule: "first" / "last" positional argument, "number" (formatnum:N), "value unit" (unit conversion), "text=" parameter
INLINE_TEMPLATES = {
    "formatnum": "number", "رقم": "first", "val": "first", "صغير": "first", "small": "first", "ملا": "first",
    "بدون لف": "first", "nowrap": "first", "لا لف": "first", "فرز تاريخ": "first",
    "اللغة": "last", "رمز لغة": "last", "لغة": "last", "lang": "last", "وإو": "first", "ill-wd2": "first",
    "حول": "value unit", "حول مختصر": "value unit", "convert": "value unit", "font": "text=",
}
INNER_TEMPLATE = re.compile(r"\{\{\s*([^{}|]+?)\s*(?:\|([^{}]*))?\}\}")


def keep_inline_templates(text, stats):
    def repl(m):
        name, args = m.group(1).strip(), m.group(2)
        low = name.lower()
        if low.startswith("formatnum:"):
            stats["inline_templates_kept"] += 1
            return name.split(":", 1)[1].strip()
        rule = INLINE_TEMPLATES.get(low) or ("last" if low.startswith("lang-") else None)
        if not rule or args is None:
            return m.group(0)
        parts = [a.strip() for a in args.split("|")]
        positional = [a for a in parts if "=" not in a]
        named = dict(a.split("=", 1) for a in parts if "=" in a)
        if rule == "text=":
            value = named.get("text", "")
        elif rule == "value unit":
            value = " ".join(positional[:2])
        elif not positional:
            return m.group(0)
        else:
            value = positional[0] if rule == "first" else positional[-1]
        stats["inline_templates_kept"] += 1
        return value

    return INNER_TEMPLATE.sub(repl, text)


def mediawiki_to_text(wikitext, stats, template_names=None):
    categories = []
    t = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    for tag in REMOVE_BLOCKS:
        t = re.sub(rf"<{tag}\b[^>/]*/>", "", t, flags=re.I)
        t = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}\s*>", "", t, flags=re.I | re.S)
    t = keep_inline_templates(t, stats)
    t = remove_nested(t, "{{", "}}", stats, "template", template_names)
    t = remove_nested(t, "{|", "|}", stats, "table")
    t = replace_links(t, categories)
    t = re.sub(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]", r"\1", t)  # [url label] -> label
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"</?[a-zA-Z][^<>]*>", "", t)  # remaining HTML tags; inner text stays
    t = re.sub(r"__[A-Zا-ي]+__", "", t)
    t = t.replace("'''", "").replace("''", "")
    t = html.unescape(html.unescape(t))  # wikitext also contains entities such as &nbsp;
    stats["presentation_forms"] += len(PRESENTATION_FORM.findall(t))
    t = PRESENTATION_FORM.sub(lambda m: unicodedata.normalize("NFKC", m.group()), t)

    lines, drop_level = [], None
    for line in t.split("\n"):
        s = line.strip()
        heading = re.match(r"^(=+)\s*(.*?)\s*=+$", s)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            if drop_level is not None and level <= drop_level:
                drop_level = None
            if title.lower().rstrip(":") in DROP_SECTIONS:
                drop_level = level
                stats["sections_dropped"] += 1
                continue
            if drop_level is None and title:
                lines.append("")
                lines.append(HEADING + title)
            continue
        if drop_level is not None:
            continue
        if re.match(r"^[*#:;]+", s):  # list item: keep only if it reads like a sentence
            s = re.sub(r"^[*#:;]+\s*", "", s)
            if len(s.split()) < 5:
                stats["short_list_items_dropped"] += 1
                continue
        if s.startswith(("|", "!", "{", "}")) or any(x in s for x in ("{{", "}}", "[[", "]]", "{|", "|}")):
            stats["broken_markup_lines"] += 1
            continue
        lines.append(s)
    # a heading followed by another heading or by nothing had all its content removed: drop it
    kept = []
    for i, line in enumerate(lines):
        if line.startswith(HEADING):
            nxt = next((l for l in lines[i + 1:] if l.strip()), None)
            if nxt is None or nxt.startswith(HEADING):
                stats["empty_headings_dropped"] += 1
                continue
            line = line[len(HEADING):]
        kept.append(line)
    text = clean_text("\n".join(kept))
    text = re.sub(r"\(\s*[،,;]?\s*\)", "", text)  # "( )" left by removed templates
    text = re.sub(r"[ \t]+([،,.:؛])", r"\1", text)
    return clean_text(text), categories
