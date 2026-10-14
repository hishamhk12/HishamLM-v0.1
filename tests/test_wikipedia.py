import unittest
from collections import Counter

from data.wikipedia_clean import mediawiki_to_text
from data.wikipedia_extract import pages

XML = """<mediawiki>
  <page>
    <title>قطة</title>
    <ns>0</ns>
    <id>7</id>
    <revision>
      <id>100</id>
      <timestamp>2020-01-01T00:00:00Z</timestamp>
      <contributor>
        <username>a</username>
        <id>55</id>
      </contributor>
      <text bytes="10" xml:space="preserve">نسخة قديمة</text>
    </revision>
    <revision>
      <id>200</id>
      <timestamp>2026-01-01T00:00:00Z</timestamp>
      <contributor>
        <username>b</username>
        <id>66</id>
      </contributor>
      <text bytes="30" xml:space="preserve">السطر الأول
السطر &amp; الثاني</text>
    </revision>
  </page>
  <page>
    <title>هر</title>
    <ns>0</ns>
    <id>8</id>
    <redirect title="قطة" />
    <revision>
      <id>300</id>
      <timestamp>2021-01-01T00:00:00Z</timestamp>
      <text bytes="20" xml:space="preserve">#تحويل [[قطة]]</text>
    </revision>
  </page>
</mediawiki>
"""


class TestWikipediaExtract(unittest.TestCase):
    def test_latest_revision_and_redirect(self):
        result = list(pages(line + "\n" for line in XML.split("\n")))
        self.assertEqual(len(result), 2)
        cat, redirect = result
        self.assertEqual((cat["id"], cat["ns"], cat["title"], cat["revisions"]), (7, 0, "قطة", 2))
        self.assertEqual(cat["rev_id"], 200)  # not the contributor id
        self.assertEqual(cat["timestamp"], "2026-01-01T00:00:00Z")
        self.assertEqual(cat["text"], "السطر الأول\nالسطر &amp; الثاني")  # unescaped later, in extract()
        self.assertEqual(redirect["redirect"], "قطة")


class TestMediaWikiToText(unittest.TestCase):
    def test_markup_removed_text_kept(self):
        wikitext = (
            "{{صندوق معلومات مدينة|الاسم = س|السكان = {{formatnum:12}}}}\n"
            "'''س''' ({{اللغة|it|Esse}}) [[مدينة|مدينةٌ]] في [[إيطاليا]]، سكانها {{formatnum:3914}} نسمة.<ref>{{استشهاد ويب|مسار=x}}</ref>\n"
            "[[ملف:س.jpg|تصغير|صورة [[س]]]]\n"
            "{| class=\"wikitable\"\n| خلية\n|}\n"
            "== التاريخ ==\n"
            "أسست في [[القرن 12]] وفق [https://example.org مصدر].\n"
            "* عنصر قصير\n"
            "== المراجع ==\n"
            "* مرجع طويل جدًا لا يجب أن يبقى في النص أبدًا\n"
            "[[تصنيف:مدن إيطاليا]]\n[[en:Esse]]\n"
        )
        stats = Counter()
        text, categories = mediawiki_to_text(wikitext, stats)
        self.assertEqual(text, "س (Esse) مدينةٌ في إيطاليا، سكانها 3914 نسمة.\n\nالتاريخ\nأسست في القرن 12 وفق مصدر.")
        self.assertEqual(categories, ["مدن إيطاليا"])
        self.assertEqual(stats["sections_dropped"], 1)
        self.assertEqual(stats["table_count"], 1)



class TestWikiStyle(unittest.TestCase):
    def test_markers_and_patterns(self):
        import re

        from inference.wiki_style_eval import PATTERNS, arabic_words, marker_rate

        text = "خربة الفرس هي قرية تقع في محافظة طرطوس، ويُعد نهرها من أجمل الأنهار. وُلد في المدينة شاعر (مواليد 1950)."
        ws = arabic_words(text)
        self.assertAlmostEqual(marker_rate(ws), 100 * 4 / len(ws))  # هي، تقع، ويُعد، وُلد
        hits = sum(len(re.findall(p, text.replace("ُ", ""))) for p in PATTERNS.values())
        self.assertEqual(hits, 4)  # (مواليد، هي قرية، تقع في، ولد في


if __name__ == "__main__":
    unittest.main()
