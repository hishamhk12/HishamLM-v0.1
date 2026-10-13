import tempfile
import unittest
import zipfile
from pathlib import Path

from data.prepare import PrepareConfig, read_epub
from data.sources import PRIVATE_SOURCES, SOURCES

OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf"><manifest>
  <item id="c" href="Content/cover.xhtml"/><item id="r" href="Content/copyright.xhtml"/>
  <item id="ch2" href="Content/chapter-2.xhtml"/><item id="ch1" href="Content/chapter-1.xhtml"/>
  <item id="refs" href="Content/chapter-3.xhtml"/>
</manifest><spine><itemref idref="c"/><itemref idref="r"/><itemref idref="ch1"/><itemref idref="ch2"/><itemref idref="refs"/></spine></package>"""
CONTAINER = """<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="EPUB/package.opf"/></rootfiles></container>"""


def page(body):
    return f"<html><body>{body}</body></html>"


class TestEpub(unittest.TestCase):
    def test_read_epub(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "book.epub"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("META-INF/container.xml", CONTAINER)
                z.writestr("EPUB/package.opf", OPF)
                z.writestr("EPUB/Content/cover.xhtml", page("<p>غلاف</p>"))
                z.writestr("EPUB/Content/copyright.xhtml", page("<p>جميع الحقوق محفوظة</p>"))
                z.writestr("EPUB/Content/chapter-1.xhtml", page("<h1>كتاب تجريبي</h1><p>الفقرة الأولى\n   تكملة السطر.</p><p>الفقرة الثانية.</p>"))
                z.writestr("EPUB/Content/chapter-2.xhtml", page("<h1>كتاب تجريبي</h1><p>الفصل الثاني.</p>"))
                z.writestr("EPUB/Content/chapter-3.xhtml", page("<h1>كتاب تجريبي</h1><p>Anderson, Chris. 2008. The End of Theory. Wired.</p>"))
            text, skipped = read_epub(path, "كتاب تجريبي")
        self.assertEqual([l for l in text.split("\n") if l.strip()],
                         ["الفقرة الأولى تكملة السطر.", "الفقرة الثانية.", "الفصل الثاني."])
        self.assertEqual(skipped, ["Content/cover.xhtml", "Content/copyright.xhtml", "Content/chapter-3.xhtml"])

    def test_private_sources_are_opt_in(self):
        self.assertFalse(any(s.get("private") for s in SOURCES))
        self.assertTrue(all(s.get("private") for s in PRIVATE_SOURCES))
        self.assertFalse(PrepareConfig().include_private)


if __name__ == "__main__":
    unittest.main()
