import tempfile
import unittest
from pathlib import Path

from tokenizer.char_tokenizer import CharTokenizer


class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.text = "hello world\nمرحبا بالعالم"
        self.tok = CharTokenizer.from_text(self.text)

    def test_vocab(self):
        # every distinct character plus the <unk> token
        self.assertEqual(self.tok.vocab_size, len(set(self.text)) + 1)
        self.assertEqual(self.tok.itos[CharTokenizer.UNK_ID], CharTokenizer.UNK_TOKEN)
        for ch in set(self.text):
            self.assertEqual(self.tok.itos[self.tok.stoi[ch]], ch)

    def test_roundtrip(self):
        ids = self.tok.encode(self.text)
        self.assertEqual(len(ids), len(self.text))
        self.assertEqual(self.tok.decode(ids), self.text)

    def test_unknown_characters(self):
        ids = self.tok.encode("hello!")
        self.assertEqual(ids[-1], CharTokenizer.UNK_ID)
        self.assertEqual(self.tok.decode(ids), "hello" + CharTokenizer.UNK_TOKEN)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vocab.json"
            self.tok.save(path)
            loaded = CharTokenizer.load(path)
        self.assertEqual(loaded.itos, self.tok.itos)
        self.assertEqual(loaded.encode(self.text), self.tok.encode(self.text))


if __name__ == "__main__":
    unittest.main()
