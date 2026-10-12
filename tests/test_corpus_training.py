import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from data.prepare import PrepareConfig, prepare
from inference.memorization import longest_match, ngram_overlap
from model.transformer_lm import TransformerLanguageModel
from tests.test_prepare import write_fixture
from training.config import TrainConfig
from training.data import TextData
from training.evaluate import perplexity, stream_loss
from training.train import learning_rate, train


class TestPieces(unittest.TestCase):
    def test_learning_rate_schedule(self):
        cfg = TrainConfig(learning_rate=1e-3, lr_schedule="cosine", warmup_steps=10, min_lr=1e-4, max_steps=110)
        self.assertAlmostEqual(learning_rate(5, cfg), 5e-4)
        self.assertAlmostEqual(learning_rate(10, cfg), 1e-3)
        self.assertAlmostEqual(learning_rate(60, cfg), 5.5e-4)
        self.assertAlmostEqual(learning_rate(110, cfg), 1e-4)
        self.assertEqual(learning_rate(50, TrainConfig(learning_rate=3e-3)), 3e-3)  # constant

    def test_weighted_sampling(self):
        a, b = torch.zeros(1000, dtype=torch.long), torch.ones(1000, dtype=torch.long)
        data = TextData(None, {"a": a, "b": b}, {"a": 0.2, "b": 0.8}, {}, {})
        x1, _ = data.train_batch(4000, 8, torch.Generator().manual_seed(0))
        x2, _ = data.train_batch(4000, 8, torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(x1, x2))
        self.assertAlmostEqual(x1[:, 0].float().mean().item(), 0.8, delta=0.03)

    def test_stream_loss_scores_every_target_once(self):
        torch.manual_seed(0)
        model = TransformerLanguageModel(10, context_length=8, embedding_dim=16, n_heads=2, n_layers=1).eval()
        for n in (5, 9, 30, 31):
            for stride in (1, 3, 8):
                _, count = stream_loss(model, torch.randint(10, (n,)), 8, stride)
                self.assertEqual(count, n - 1)
        # with stride == context_length the windows do not overlap: compare to a direct computation
        ids = torch.randint(10, (25,))
        loss, _ = stream_loss(model, ids, 8, 8)
        with torch.no_grad():
            nll = [F.cross_entropy(model(ids[None, s : s + 8])[0], ids[s + 1 : s + 9], reduction="sum") for s in (0, 8)]
            nll.append(F.cross_entropy(model(ids[None, 16:24])[0], ids[17:25], reduction="sum"))
        self.assertAlmostEqual(loss, sum(t.item() for t in nll) / 24, places=5)

    def test_perplexity_is_safe(self):
        self.assertAlmostEqual(perplexity(math.log(5.0)), 5.0)
        self.assertEqual(perplexity(1000.0), float("inf"))

    def test_memorization_measures(self):
        corpus = "the quick brown fox jumps over the lazy dog"
        self.assertEqual(longest_match("a brown fox jumps high", corpus), (17, " brown fox jumps "))
        self.assertEqual(ngram_overlap(corpus, corpus, n=8, step=1), 1.0)
        self.assertEqual(ngram_overlap("zzzzzzzzzzzzzzzzzz", corpus, n=8), 0.0)


class TestCorpusTraining(unittest.TestCase):
    def test_train_on_prepared_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_fixture(root)
            source = {"name": "ntrex_ar", "language": "ar", "url": "x/ntrex_ar.txt", "license": "test",
                      "license_url": "", "license_note": "", "repository": "", "description": ""}
            pcfg = PrepareConfig(out_dir=str(root / "ds"), domain_dir=str(root / "domain"),
                                 eval_questions=str(root / "questions.jsonl"), block_chars=500, domain_max_doc_share=0.2)
            with mock.patch("data.prepare.source_path", lambda s: root / "raw" / "ntrex_ar.txt"):
                dmeta = prepare(pcfg, sources=[source], verify_hashes=False)

            # learning_rate 0: the model never improves, so early stopping must end the run
            cfg = TrainConfig(run_name="run", model_type="transformer", dataset=str(root / "ds"), device="cpu",
                              checkpoint_dir=str(root / "ckpt"), context_length=8, embedding_dim=16, n_heads=2,
                              n_layers=1, batch_size=4, learning_rate=0.0, max_steps=50, eval_interval=2,
                              eval_batches=2, early_stopping_patience=3)
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(cfg)
            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["history"][-1]["step"], 8)  # best at 2, then 3 evaluations without improvement

            run_dir = root / "ckpt" / "run"
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            for key in ("seed", "config", "git_commit", "dataset_fingerprint", "vocab_size", "num_parameters",
                        "torch_version", "python_version", "best_step", "training_seconds"):
                self.assertIn(key, meta)
            self.assertEqual(meta["dataset_fingerprint"], dmeta["dataset_fingerprint"])
            self.assertEqual(torch.load(run_dir / "best.pt")["run_meta"]["seed"], cfg.seed)
            header = (run_dir / "log.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, "step,train_loss,val_loss,A_general_val,B_domain_val,learning_rate,elapsed_time")


if __name__ == "__main__":
    unittest.main()
