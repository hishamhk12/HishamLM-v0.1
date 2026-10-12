import contextlib
import io
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from data.prepare import PrepareConfig, prepare
from tests.test_prepare import write_fixture
from tokenizer import load_tokenizer, tokenizer_from_dict
from tokenizer.bpe_tokenizer import BPETokenizer, pretokenize
from training.config import TrainConfig
from training.data import prepared_data
from training.evaluate import stream_loss
from training.train import learning_rate, train

TRAIN = ["the cat sat on the mat. the cat ate the rat.", "القط جلس على السجادة. القط أكل الطعام في البيت."]
ALPHABET = "ab ال قطxyz.،؟!\n\t0123456789ًٌُـ😀ž_-«»'\" "


class TestBPE(unittest.TestCase):
    def setUp(self):
        self.tok = BPETokenizer.train(TRAIN, vocab_size=256 + 30 + 40, min_char_freq=1)

    def test_pretokenize_covers_every_character(self):
        rng = random.Random(0)
        for _ in range(200):
            text = "".join(rng.choice(ALPHABET) for _ in range(rng.randrange(60)))
            self.assertEqual("".join(pretokenize(text)), text)

    def test_round_trip_and_byte_fallback(self):
        rng = random.Random(1)
        samples = TRAIN + ["unseen ž 😀 漢字 \x00 end", ""] + ["".join(rng.choice(ALPHABET) for _ in range(80)) for _ in range(50)]
        for text in samples:
            ids = self.tok.encode(text)
            self.assertEqual(self.tok.decode(ids), text)
            self.assertEqual(sum(self.tok.token_char_lengths()[i] for i in ids), len(text))
        self.assertTrue(all(i < 256 for i in self.tok.encode("漢")))  # unseen character -> its UTF-8 bytes
        self.assertEqual(len(self.tok.encode("漢")), 3)

    def test_vocab_size_and_determinism(self):
        self.assertEqual(self.tok.vocab_size, 256 + len(self.tok.chars) + len(self.tok.merges))
        again = BPETokenizer.train(TRAIN, vocab_size=self.tok.vocab_size, min_char_freq=1)
        self.assertEqual(again.merges, self.tok.merges)
        self.assertEqual(again.encode(TRAIN[1]), self.tok.encode(TRAIN[1]))

    def test_merges_come_from_training_text_only(self):
        merged = {self.tok.token_text(i) for i in range(256 + len(self.tok.chars), self.tok.vocab_size)}
        for piece in merged:
            self.assertTrue(any(piece in t for t in TRAIN), piece)
        self.assertNotIn(" the", BPETokenizer.train(["القط القط القط"], 300, 1).to_dict()["chars"])

    def test_weights_change_the_merges(self):
        en_first = BPETokenizer.train(TRAIN, 256 + 30 + 1, 1, weights=[10.0, 1.0])
        ar_first = BPETokenizer.train(TRAIN, 256 + 30 + 1, 1, weights=[1.0, 10.0])
        self.assertTrue(en_first.token_text(en_first.vocab_size - 1).isascii())
        self.assertFalse(ar_first.token_text(ar_first.vocab_size - 1).isascii())

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            self.tok.save(Path(d) / "bpe.json", extra={"merges_count": 1})
            loaded = load_tokenizer(Path(d) / "bpe.json")
        self.assertEqual(loaded.merges, self.tok.merges)
        self.assertEqual(tokenizer_from_dict(self.tok.to_dict()).encode(TRAIN[0]), self.tok.encode(TRAIN[0]))


class TestMetricsAndSchedule(unittest.TestCase):
    def test_bpc_equals_loss_over_ln2_for_characters(self):
        from model.transformer_lm import TransformerLanguageModel
        torch.manual_seed(0)
        model = TransformerLanguageModel(10, 8, 16, 2, 1).eval()
        loss, count, chars = stream_loss(model, torch.randint(10, (40,)), 8, 4, torch.ones(10))
        self.assertEqual(count, chars)
        two_chars = torch.full((10,), 2.0)  # tokens that stand for 2 characters each: half the BPC
        loss2, count2, chars2 = stream_loss(model, torch.randint(10, (40,)), 8, 4, two_chars)
        self.assertEqual(chars2, 2 * count2)

    def test_lr_holds_after_decay(self):
        cfg = TrainConfig(learning_rate=1e-3, lr_schedule="cosine", warmup_steps=10, min_lr=1e-4, max_steps=100,
                          lr_decay_steps=50)
        self.assertAlmostEqual(learning_rate(50, cfg), 1e-4)
        self.assertAlmostEqual(learning_rate(99, cfg), 1e-4)
        short = TrainConfig(learning_rate=1e-3, lr_schedule="cosine", warmup_steps=10, min_lr=1e-4, max_steps=50)
        for step in range(51):  # identical to a run that ended at the decay step
            self.assertAlmostEqual(learning_rate(step, cfg), learning_rate(step, short))


class TestBPETraining(unittest.TestCase):
    def test_bpe_run_resume_and_domain_share(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_fixture(root)
            source = {"name": "ntrex_ar", "language": "ar", "url": "x/ntrex_ar.txt", "license": "test",
                      "license_url": "", "license_note": "", "repository": "", "description": ""}
            pcfg = PrepareConfig(out_dir=str(root / "ds"), domain_dir=str(root / "domain"),
                                 eval_questions=str(root / "questions.jsonl"), block_chars=500, domain_max_doc_share=0.2)
            with mock.patch("data.prepare.source_path", lambda s: root / "raw" / "ntrex_ar.txt"):
                prepare(pcfg, sources=[source], verify_hashes=False)
            texts = [p.read_text(encoding="utf-8") for p in sorted((root / "ds" / "train").glob("*.txt"))]
            BPETokenizer.train(texts, 256 + 120, min_char_freq=1).save(root / "bpe.json")

            data = prepared_data(root / "ds", tokenizer_path=root / "bpe.json", domain_share=0.1)
            self.assertAlmostEqual(data.weights.sum().item(), 1.0, places=5)
            self.assertAlmostEqual(data.weights[data.names.index("after_sales")].item(), 0.1, places=5)
            self.assertIn("A_ar_val", data.evals)

            def cfg(run, steps):
                return TrainConfig(run_name=run, model_type="transformer", dataset=str(root / "ds"),
                                   tokenizer=str(root / "bpe.json"), device="cpu", checkpoint_dir=str(root / "ckpt"),
                                   context_length=8, embedding_dim=16, n_heads=2, n_layers=1, dropout=0.1,
                                   batch_size=4, learning_rate=1e-3, max_steps=steps, eval_interval=3, eval_batches=2)

            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg("straight", 6))
                train(cfg("resumed", 3))
                train(cfg("resumed", 6), resume=True)
            a = torch.load(root / "ckpt" / "straight" / "latest.pt")
            b = torch.load(root / "ckpt" / "resumed" / "latest.pt")
            for k in a["model_state"]:
                self.assertTrue(torch.equal(a["model_state"][k], b["model_state"][k]), k)
            self.assertEqual(a["tokens_seen"], 6 * 4 * 8)
            self.assertEqual(a["chars_seen"], b["chars_seen"])
            self.assertGreater(a["chars_seen"], a["tokens_seen"])  # BPE tokens stand for more than one character
            self.assertEqual(a["run_meta"]["tokenizer"], "BPETokenizer")

    def test_resume_checkpoint_from_before_bpc(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "raw").mkdir()
            (root / "raw" / "a.txt").write_text("the cat sat on the mat. the dog sat on the rug.\n" * 20, encoding="utf-8")
            cfg = TrainConfig(run_name="old", model_type="transformer", device="cpu", data_dir=str(root / "raw"),
                              checkpoint_dir=str(root / "ckpt"), context_length=8, embedding_dim=16, n_heads=2,
                              n_layers=1, batch_size=4, max_steps=3, eval_interval=3, eval_batches=2)
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg)
            path = root / "ckpt" / "old" / "latest.pt"
            ckpt = torch.load(path)
            ckpt["best_val_loss"] = ckpt.pop("best_val_bpc") * math.log(2)  # what a phase 3 checkpoint looked like
            for key in ("val_metric", "tokens_seen", "chars_seen"):
                ckpt.pop(key)
            torch.save(ckpt, path)
            cfg.max_steps = 6
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(cfg, resume=True)
            self.assertEqual(torch.load(path)["tokens_seen"], 6 * 4 * 8)
            self.assertLess(result["best_val_bpc"], 20)


if __name__ == "__main__":
    unittest.main()
