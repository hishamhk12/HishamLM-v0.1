import unittest
from collections import Counter

from data.books_candidate import AR_WORD, WORD, is_listing, normalize_chars, reversed_line, section_to_text


class TestBooksCandidate(unittest.TestCase):
    def test_words_keep_diacritics(self):
        self.assertEqual(AR_WORD.findall("مِنْ بَعْدِ ذلك"), ["مِنْ", "بَعْدِ", "ذلك"])
        self.assertEqual(WORD.findall("كتابٌ the 12 cat_x"), ["كتابٌ", "the", "cat", "x"])

    def test_ocr_letter_shapes_are_normalized_and_counted(self):
        counts = Counter()
        self.assertEqual(normalize_chars("ڪان یوم ۲٠", counts), "كان يوم ٢٠")
        self.assertEqual(counts["ڪ->ك"], 1)
        self.assertEqual(counts["ی->ي"], 1)

    def test_presentation_forms_become_letters(self):
        self.assertEqual(normalize_chars("ﻻ", Counter()), "لا")

    def test_reversed_line(self):
        self.assertTrue(reversed_line("باتكلا يف ةءارقلا نم ريثكلا دجن انه"))
        self.assertFalse(reversed_line("نجد هنا الكثير من القراءة في الكتاب"))

    def test_listing(self):
        self.assertTrue(is_listing("ملكة الجمال\nشعر جديد\nثم ماذا؟\nالقصص\nالكتب"))
        self.assertFalse(is_listing("كان الرجل يمشي في الطريق الطويل حين رأى صديقه القديم، فوقف وسلم عليه بحرارة.\nثم مضيا معًا."))

    def test_section_to_text_drops_head_alt_and_footnote_numbers(self):
        html = ('<html><head><title>Page 3</title></head><body><img alt="غلاف"/>'
                '<p>نص<a class="noteref" href="n.html#1">1</a> الكتاب</p></body></html>')
        self.assertEqual(" ".join(section_to_text(html).split()), "نص الكتاب")


if __name__ == "__main__":
    unittest.main()
