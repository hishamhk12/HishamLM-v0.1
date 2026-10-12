import json
import tempfile
import unittest
from pathlib import Path

from data.clean import clean_text
from data.extract_repo import SECRET_PATTERNS, code_to_text, html_to_text, js_string_literals, redact, run

DOMAIN_AR = "خدمة التوصيل تبدأ عند وصول الفاتورة من SAP إلى Odoo، ثم يحجز العميل موعد التركيب مع الفني."
DOMAIN_EN = "The customer books a delivery appointment after the invoice arrives, then the technician installs."


class TestClean(unittest.TestCase):
    def test_clean_text(self):
        raw = "\ufeffمرحبا\u200f  بالعالم\r\nline\x00 two\t\tend   \r\n\r\n\r\n\r\nلا تُغيَّر الحروف: أإآ ة ى ـ"
        self.assertEqual(clean_text(raw), "مرحبا بالعالم\nline two end\n\nلا تُغيَّر الحروف: أإآ ة ى ـ")


class TestExtractors(unittest.TestCase):
    def test_js_literals(self):
        src = '''// a comment with "quotes"
        const a = "first string"; /* "skipped" */
        const b = `قبل ${term("جديد", "New")} بعد`;
        const c = `<p>${items.map(x => `<li>${x}</li>`).join("")}</p>`;'''
        literals = js_string_literals(src)
        self.assertEqual(literals[0], "first string")
        self.assertIn("قبل جديد (New) بعد", literals)
        self.assertNotIn("a comment with", " ".join(literals))

    def test_code_keeps_only_prose(self):
        src = f'''const x = {{ cls: "panel invoice-training-section delivery-request", text: "{DOMAIN_EN}", id: "BR-01" }};'''
        self.assertEqual(code_to_text(src), DOMAIN_EN)

    def test_html(self):
        html = '<style>p{color:red}</style><p>أول فقرة</p><script>var k="x"</script><img alt="وصف الصورة"><p>ثاني</p>'
        self.assertEqual(clean_text(html_to_text(html)), "أول فقرة\n\nوصف الصورة\n\nثاني")

    def test_redaction(self):
        text, counts = redact("see https://x.dev/a?b=1 or C:\\Users\\me\\f.md mail a.b@c.com call 0551234567 invoice 4213030704 year 2026")
        self.assertEqual(text, "see [URL] or [PATH] mail [EMAIL] call [PHONE] invoice [ID] year 2026")
        self.assertEqual(counts, {"url": 1, "path": 1, "email": 1, "phone": 1, "long_number": 1})

    def test_secret_patterns(self):
        self.assertTrue(SECRET_PATTERNS.search('const key = "sk-abcdefghijklmnopqrstuvwx";'))
        self.assertTrue(SECRET_PATTERNS.search("password: 'hunter2hunter2'"))
        self.assertFalse(SECRET_PATTERNS.search('class="sub-task-cycle-completion-screenshot"'))
        self.assertFalse(SECRET_PATTERNS.search("password: true"))


class TestRun(unittest.TestCase):
    def test_repo_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            repo, out = Path(d) / "repo", Path(d) / "out"
            files = {
                "README.md": f"# دليل\n\n{DOMAIN_AR}\n\n{DOMAIN_EN}\n\n{DOMAIN_AR}\n" * 3,
                "docs/copy.md": f"# دليل\n\n{DOMAIN_AR}\n\n{DOMAIN_EN}\n\n{DOMAIN_AR}\n" * 3,  # identical file
                "docs/other.md": DOMAIN_AR + "".join(f"\n\nالحالة {i}: خدمة الصيانة تحدد موعد الفني مع العميل بعد مراجعة الفاتورة في المستودع." for i in range(4)),
                "design.md": "A calm white canvas with violet accents and generous spacing everywhere. " * 10,
                "node_modules/pkg/README.md": DOMAIN_EN * 5,
                ".env": "API_KEY=abc",
                "config.js": f'const apiKey = "sk-abcdefghijklmnopqrstuvwxyz123456"; const t = "{DOMAIN_EN}";' * 5,
                "evaluationDataset.js": f'const q = "{DOMAIN_EN}";' * 5,
                "logo.png": "not really a png",
                "package-lock.json": "{}",
            }
            for name, content in files.items():
                (repo / name).parent.mkdir(parents=True, exist_ok=True)
                (repo / name).write_text(content, encoding="utf-8")

            meta = run(repo, out)
            status = {r["path"]: (r["status"], r["reason"]) for r in meta["files"]}

            self.assertEqual(status["README.md"][0], "used")
            self.assertIn("duplicate", status["docs/copy.md"][1])
            self.assertEqual(status["docs/other.md"][0], "used")
            self.assertIn("off-domain", status["design.md"][1])
            self.assertIn("dependency", status["node_modules/pkg/README.md"][1])
            self.assertIn("credentials", status[".env"][1])
            self.assertIn("secret", status["config.js"][1])
            self.assertIn("evaluation", status["evaluationDataset.js"][1])
            self.assertEqual(status["logo.png"][0], "excluded")
            self.assertEqual(status["package-lock.json"][1], "lock file")

            corpus = (out / "corpus.txt").read_text(encoding="utf-8")
            self.assertEqual(corpus.count(DOMAIN_AR), 1)  # repeated lines removed across files
            used = [r for r in meta["files"] if r["status"] == "used"]
            for r in used:
                self.assertTrue((out / r["output"]).exists())
                self.assertGreater(r["extracted_chars"], 0)
                self.assertIn("type", r)
            self.assertEqual(meta["summary"]["files_scanned"], len(files))
            self.assertEqual(json.loads((out / "metadata.json").read_text(encoding="utf-8"))["summary"], meta["summary"])


if __name__ == "__main__":
    unittest.main()
