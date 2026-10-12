"""Does generated text copy the training corpus word for word?"""


def longest_match(text, corpus):
    """Longest substring of `text` that appears verbatim in `corpus`. Returns (length, substring).

    For each start position we only try to beat the best length found so far,
    so the number of `in` searches stays about 2 * len(text).
    """
    best, best_at = 0, 0
    for i in range(len(text)):
        while i + best < len(text) and text[i : i + best + 1] in corpus:
            best += 1
            best_at = i
    return best, text[best_at : best_at + best]


def ngram_overlap(text, corpus, n=32, step=4):
    """Fraction of the n-character windows of `text` (every `step` characters) found verbatim in `corpus`."""
    windows = [text[i : i + n] for i in range(0, len(text) - n + 1, step)]
    if not windows:
        return 0.0
    return sum(w in corpus for w in windows) / len(windows)


def memorization_report(text, corpus, n=32):
    length, match = longest_match(text, corpus)
    return {"longest_match": length, "longest_match_text": match, f"overlap_{n}gram": ngram_overlap(text, corpus, n)}


def is_copying(report, n=32, max_match=80, max_overlap=0.5):
    """Heuristic warning: a long verbatim run, or most n-grams copied from training text."""
    return report["longest_match"] >= max_match or report[f"overlap_{n}gram"] >= max_overlap
