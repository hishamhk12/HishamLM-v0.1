import unittest

from inference.coherence import content_words, text_measures
from inference.memorization import ngram_overlap
from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.context_coverage import window_stats

LEXICON = {"the", "cat", "sat", "on", "mat", "dog", "ran"}


class TestCoherenceMeasures(unittest.TestCase):
    def test_repetition_is_detected(self):
        varied = text_measures("", "The cat sat on the mat. The dog ran.", LEXICON)
        repeated = text_measures("", "The cat sat on the mat. The cat sat on the mat.", LEXICON)
        self.assertEqual(varied["repetition_4gram"], 0.0)
        self.assertGreater(repeated["repetition_4gram"], 0.0)
        self.assertGreater(repeated["template_repeat"], 0.0)
        self.assertLess(repeated["distinct_2"], varied["distinct_2"])

    def test_invented_words_and_topic_terms(self):
        r = text_measures("A cat story.", "The cat sat on the blorp. The dog ran.", LEXICON, topic_terms=["cat"])
        self.assertAlmostEqual(r["invented_word_rate"], 1 / 9)
        self.assertGreater(r["topic_term_rate"], 0)
        self.assertEqual(r["topic_in_last_third"], 0.0)  # the cat is only mentioned at the start

    def test_arabic_prefixes_and_diacritics(self):
        self.assertEqual(content_words("المكتبةُ"), content_words("والمكتبة"))
        self.assertIn("طلاب", content_words("ذهب الطلاب"))

    def test_50_char_overlap(self):
        corpus = "x" * 10 + "a long sentence that certainly appears inside the training corpus, verbatim." + "y" * 10
        self.assertEqual(ngram_overlap(corpus[10:70], corpus, 50, 5), 1.0)
        self.assertEqual(ngram_overlap("z" * 60, corpus, 50, 5), 0.0)


class TestContextCoverage(unittest.TestCase):
    def test_longer_windows_cover_more_text(self):
        text = "the cat sat on the mat. the dog ran home.\n" * 40
        tok = BPETokenizer.train([text], vocab_size=256 + 20 + 10, min_char_freq=1)
        ids = tok.encode(text)
        short, long = window_stats(tok, ids, 8), window_stats(tok, ids, 16)
        self.assertGreater(long["chars"], short["chars"])
        self.assertAlmostEqual(long["words"], 2 * short["words"], delta=1.0)


if __name__ == "__main__":
    unittest.main()


class TestVocabDrift(unittest.TestCase):
    def test_distinctive_words_and_rate(self):
        from collections import Counter

        from inference.vocab_drift import distinctive, rate

        target = Counter({"بيانات": 50, "كتاب": 50})
        background = Counter({"كتاب": 500, "بيت": 500})
        self.assertEqual(distinctive(target, background, min_count=30, ratio=8, top=10), ["بيانات"])
        self.assertAlmostEqual(rate("البيانات كثيرة في الكتاب", {"بيانات"}), 100 / 3)
