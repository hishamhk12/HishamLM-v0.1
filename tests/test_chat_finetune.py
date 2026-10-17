import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from inference.generate import load_checkpoint
from training.chat_finetune import encode_example, finetune, prompt_ids, windows
from training.config import TrainConfig
from training.train import train

TEXT = "<USER>\nhi there\n<ASSISTANT>\nthe cat sat on the mat.\n" * 30


class CharTok:
    def encode(self, s):
        return [ord(c) for c in s]


class TestChatFinetune(unittest.TestCase):
    def test_mask_only_last_assistant(self):
        msgs = [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cd"},
                {"role": "user", "content": "ef"}, {"role": "assistant", "content": "gh"}]
        ids, mask = encode_example(CharTok(), msgs)
        self.assertEqual("".join(chr(i) for i in ids), "<USER>\nab\n<ASSISTANT>\ncd\n<USER>\nef\n<ASSISTANT>\ngh\n")
        self.assertEqual("".join(chr(i) for i, m in zip(ids, mask) if m), "gh")

    def test_prompt_matches_training_prefix(self):
        msgs = [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cd"}]
        ids, mask = encode_example(CharTok(), msgs)
        p = prompt_ids(CharTok(), msgs[:1], 4)
        self.assertEqual(p, ids[: mask.index(True)])
        self.assertEqual(prompt_ids(CharTok(), msgs[:1], 40)[:3], [ord("\n")] * 3)  # left padding

    def test_windows_score_each_target_once(self):
        T, stride = 8, 4
        ids = list(range(100, 130))
        mask = [False] * 5 + [True] * 20 + [False] * 5
        seen = []
        for x, y, m in windows(ids, mask, T, stride, pad_id=0):
            self.assertEqual(len(x), T)
            self.assertEqual(x[1:], y[:-1])
            for k in range(T):
                if m[k]:
                    seen.append(y[k])
                    self.assertGreaterEqual(k + 1, T - stride)  # at least T - stride tokens of context
        self.assertEqual(seen, ids[5:25])

    def test_finetune_runs_and_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / "a.txt").write_text(TEXT, encoding="utf-8")
            cfg = TrainConfig(run_name="base", model_type="transformer", device="cpu", data_dir=str(root / "raw"),
                              checkpoint_dir=str(root / "ckpt"), context_length=8, embedding_dim=16, n_heads=2,
                              n_layers=1, batch_size=4, learning_rate=1e-3, max_steps=3, eval_interval=3, eval_batches=1)
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg)
            _, tok, _ = load_checkpoint(root / "ckpt/base/latest.pt", "cpu")
            msgs = [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "the cat sat on the mat."}]
            ws = windows(*encode_example(tok, msgs), 8, 4, tok.encode("\n")[0])
            data = {s: tuple(torch.tensor([w[k] for w in ws]) for k in range(3)) for s in ("train", "val")}
            with contextlib.redirect_stdout(io.StringIO()):
                meta = finetune(root / "ckpt/base/latest.pt", root / "chat", epochs=2, batch_size=2, warmup=2,
                                eval_interval=2, patience=0, stride=4, data=data)
            self.assertLessEqual(meta["best_chat_val_bpc"], meta["base_chat_val_bpc"])
            model, _, ckpt = load_checkpoint(root / "chat/best.pt", "cpu")
            self.assertEqual(ckpt["val_metric"], "chat_bpc")
            self.assertTrue((root / "chat/latest.pt").exists())


if __name__ == "__main__":
    unittest.main()
