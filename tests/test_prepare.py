import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data.prepare import (PrepareConfig, assign_splits, decontaminate, fix_leading_punctuation,
                          is_modern_arabic_sentence, make_blocks, prepare)
from tokenizer.char_tokenizer import CharTokenizer

AR_LINES = [f"الجملة رقم {i} تتحدث عن موضوع جديد في المدينة." for i in range(400)]
QUESTION = "ما هي مراحل خدمة التوصيل في النظام؟"


def write_fixture(root):
    raw = root / "raw"
    raw.mkdir()
    # an unseen character (§) only in the last lines, which end up in some split
    (raw / "ntrex_ar.txt").write_text("\n".join(AR_LINES) + "\n", encoding="utf-8")
    domain = root / "domain" / "nested"
    domain.mkdir(parents=True)
    # every line differs within any 50 characters, so held-out documents do not repeat train
    who = ["الفني", "السائق", "المشرف", "العميل", "المصمم", "المندوب", "المحاسب"]
    act = ["يراجع", "يؤكد", "يحجز", "يلغي", "يعدل"]
    what = ["الموعد", "الفاتورة", "المهمة", "الشحنة", "التذكرة", "القياسات"]
    for i in range(12):
        lines = [f"{who[(i + j) % 7]} {i}{j} {act[(i * j) % 5]} {what[(i + 2 * j) % 6]} {j}{i} في الفرع {i * 31 + j}"
                 f" ثم {who[(i * 3 + j) % 7]} {act[(j + 1) % 5]} {what[(i + j) % 6]} {i}-{j}." for j in range(30)]
        (domain / f"doc{i:02d}.txt").write_text("\n".join(lines), encoding="utf-8")
    (domain / "doc00.txt").write_text((domain / "doc00.txt").read_text(encoding="utf-8") + "\nرمز نادر §", encoding="utf-8")
    (root / "questions.jsonl").write_text(json.dumps({"question": QUESTION}, ensure_ascii=False) + "\n", encoding="utf-8")
    return raw


class TestHelpers(unittest.TestCase):
    def test_modern_arabic_filter(self):
        self.assertTrue(is_modern_arabic_sentence("أريد أن أذهب إلى اليابان."))
        self.assertFalse(is_modern_arabic_sentence("وَمَا ذَرَأَ لَكُمْ فِي الْأَرْضِ مُخْتَلِفًا أَلْوَانُهُ"))
        self.assertFalse(is_modern_arabic_sentence("خطب جسيم فتت الأكبادا"))
        self.assertEqual(fix_leading_punctuation("! أمسكه"), "أمسكه!")
        self.assertEqual(fix_leading_punctuation("كيف حالك؟"), "كيف حالك؟")

    def test_blocks_cut_at_line_boundaries(self):
        text = "\n".join(AR_LINES)
        blocks = make_blocks(text, 500)
        self.assertEqual("\n".join(blocks), text)
        self.assertTrue(all(len(b) < 600 for b in blocks))

    def test_decontaminate(self):
        train = "مقدمة.\nلا ينتقل المنتج من التصنيع إلى التركيب مباشرة لأن خدمة التوصيل تقع بينهما في كل الحالات.\nملخص الدورة"
        held_out = ("سطر جديد تمامًا عن موضوع آخر لم يظهر في التدريب.\n"
                    "ملاحظة: لا ينتقل المنتج من التصنيع إلى التركيب مباشرة لأن خدمة التوصيل تقع بينهما في كل الحالات.\n"
                    "ملخص الدورة")
        cleaned, dropped = decontaminate(held_out, train)
        self.assertEqual(cleaned, "سطر جديد تمامًا عن موضوع آخر لم يظهر في التدريب.")
        self.assertEqual(dropped, 2)

    def test_split_is_deterministic(self):
        units = [(f"u{i}", 100) for i in range(100)]
        a = assign_splits(units, 0.05, 0.05, "seed")
        self.assertEqual(a, assign_splits(units, 0.05, 0.05, "seed"))
        self.assertNotEqual(a, assign_splits(units, 0.05, 0.05, "other seed"))
        self.assertEqual(sum(s == "val" for s in a.values()), 5)
        self.assertEqual(sum(s == "test" for s in a.values()), 5)


class TestPrepare(unittest.TestCase):
    def run_prepare(self, root, out_name):
        raw = root / "raw"
        source = {"name": "ntrex_ar", "language": "ar", "url": "x/ntrex_ar.txt", "license": "test", "license_url": "",
                  "license_note": "", "repository": "", "description": ""}
        cfg = PrepareConfig(out_dir=str(root / out_name), domain_dir=str(root / "domain"),
                            eval_questions=str(root / "questions.jsonl"), block_chars=500,
                            domain_max_doc_share=0.2)
        with mock.patch("data.prepare.source_path", lambda s: raw / "ntrex_ar.txt"):
            return prepare(cfg, sources=[source], verify_hashes=False), root / out_name

    def test_prepare(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_fixture(root)
            meta, out = self.run_prepare(root, "out1")

            # recursive domain loading picked up the nested files
            self.assertEqual(sum(u["source"] == "after_sales" for u in meta["split_units"]), 12)

            # whole documents/blocks go to exactly one split, and no line is shared between splits
            domain_units = [u for u in meta["split_units"] if u["source"] == "after_sales"]
            self.assertEqual({u["split"] for u in domain_units}, {"train", "val", "test"})
            self.assertEqual(meta["leakage_checks"]["train_lines_also_in_val_or_test"], 0)
            train = "".join(p.read_text(encoding="utf-8") for p in (out / "train").glob("*.txt"))
            held_out = "".join(p.read_text(encoding="utf-8") for split in ("val", "test") for p in (out / split).glob("*.txt"))
            for line in held_out.split("\n"):
                if len(line.split()) >= 3:
                    self.assertNotIn(line, train)

            # vocabulary comes from train only; characters only seen outside train are <unk>
            vocab = CharTokenizer.load(out / "vocab.json")
            self.assertEqual(set(vocab.itos[1:]), set(train))
            doc00 = next(u for u in domain_units if u["unit"] == "doc00.txt")
            if doc00["split"] != "train":
                self.assertNotIn("§", vocab.stoi)
                self.assertGreater(meta["sources"]["after_sales"]["unk_rate"][doc00["split"]], 0)
            self.assertEqual(meta["sources"]["after_sales"]["unk_rate"]["train"], 0)

            # sampling weights sum to 1 and give the domain its planned share
            self.assertAlmostEqual(sum(meta["sampling"]["weights"].values()), 1.0, places=3)
            self.assertEqual(meta["sampling"]["weights"]["after_sales"], 0.175)

            # same inputs and seed -> same dataset fingerprint
            meta2, _ = self.run_prepare(root, "out2")
            self.assertEqual(meta["dataset_fingerprint"], meta2["dataset_fingerprint"])

    def test_dedup_across_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_fixture(root)
            shared = "الفني يحجز موعد التركيب مع العميل بعد وصول الفاتورة اليوم."
            raw_file = root / "raw" / "ntrex_ar.txt"
            raw_file.write_text(raw_file.read_text(encoding="utf-8") + shared + "\n", encoding="utf-8")
            (root / "domain" / "nested" / "doc05.txt").write_text(shared + "\n" + "سطر إضافي عن الصيانة والمستودع والفاتورة.", encoding="utf-8")
            meta, out = self.run_prepare(root, "out")
            everything = "".join(p.read_text(encoding="utf-8") for p in out.rglob("*.txt"))
            self.assertEqual(everything.count(shared), 1)
            self.assertEqual(meta["sources"]["ntrex_ar"]["duplicate_lines_removed"], 1)


if __name__ == "__main__":
    unittest.main()
