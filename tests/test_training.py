import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from inference.generate import load_checkpoint
from training.config import TrainConfig
from training.train import train

TEXT = "the cat sat on the mat. the dog sat on the rug.\n" * 20


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "raw").mkdir()
        (root / "raw" / "a.txt").write_text(TEXT, encoding="utf-8")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, run_name, max_steps, model_type="transformer"):
        return TrainConfig(
            run_name=run_name, model_type=model_type, device="cpu",
            data_dir=str(self.root / "raw"), checkpoint_dir=str(self.root / "ckpt"),
            context_length=8, embedding_dim=16, hidden_size=16, n_heads=2, n_layers=1, dropout=0.1,
            batch_size=4, learning_rate=1e-3, max_steps=max_steps, eval_interval=3, eval_batches=2,
        )

    def run_train(self, cfg, resume=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(cfg, resume=resume)

    def weights(self, run_name):
        return torch.load(self.root / "ckpt" / run_name / "latest.pt")["model_state"]

    def assertSameWeights(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_checkpoints_and_load(self):
        for model_type in ("mlp", "transformer"):
            result = self.run_train(self.config(model_type, 6, model_type))
            run_dir = self.root / "ckpt" / model_type
            self.assertTrue((run_dir / "latest.pt").exists())
            self.assertTrue((run_dir / "best.pt").exists())
            best = torch.load(run_dir / "best.pt")
            self.assertEqual(best["val_bpc"], result["best_val_bpc"])

            model, tokenizer, ckpt = load_checkpoint(run_dir / "latest.pt", "cpu")
            self.assertEqual(ckpt["step"], 6)
            self.assertSameWeights(model.state_dict(), ckpt["model_state"])
            self.assertEqual(tokenizer.decode(tokenizer.encode("the cat")), "the cat")

    def test_reproducible_with_seed(self):
        self.run_train(self.config("a", 6))
        self.run_train(self.config("b", 6))
        self.assertSameWeights(self.weights("a"), self.weights("b"))

    def test_snapshot_steps(self):
        cfg = self.config("snap", 9)
        cfg.snapshot_steps = [3, 9]
        self.run_train(cfg)
        run_dir = self.root / "ckpt" / "snap"
        self.assertEqual(sorted(p.name for p in run_dir.glob("step_*.pt")), ["step_3.pt", "step_9.pt"])
        self.assertEqual(torch.load(run_dir / "step_3.pt")["step"], 3)

    def test_resume_matches_uninterrupted_run(self):
        self.run_train(self.config("straight", 8))
        self.run_train(self.config("resumed", 4))
        self.run_train(self.config("resumed", 8), resume=True)
        self.assertEqual(torch.load(self.root / "ckpt" / "resumed" / "latest.pt")["step"], 8)
        self.assertSameWeights(self.weights("straight"), self.weights("resumed"))


if __name__ == "__main__":
    unittest.main()
