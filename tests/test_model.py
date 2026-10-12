import unittest

import torch

from model.mlp_lm import MLPLanguageModel


class TestMLPLanguageModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MLPLanguageModel(vocab_size=10, context_length=4, embedding_size=8, hidden_size=16)

    def test_output_shape(self):
        idx = torch.randint(10, (3, 7))
        self.assertEqual(self.model(idx).shape, (3, 7, 10))

    def test_causal(self):
        # changing a token must not change the predictions at earlier positions
        idx = torch.randint(10, (1, 7))
        changed = idx.clone()
        changed[0, 5] = (idx[0, 5] + 1) % 10
        a, b = self.model(idx), self.model(changed)
        self.assertTrue(torch.allclose(a[:, :5], b[:, :5]))
        self.assertFalse(torch.allclose(a[:, 5:], b[:, 5:]))

    def test_window_size(self):
        # a token more than context_length - 1 positions back is outside the window
        idx = torch.randint(10, (1, 7))
        changed = idx.clone()
        changed[0, 0] = (idx[0, 0] + 1) % 10
        a, b = self.model(idx), self.model(changed)
        self.assertFalse(torch.allclose(a[:, 3], b[:, 3]))  # position 3 sees tokens 0..3
        self.assertTrue(torch.allclose(a[:, 4:], b[:, 4:]))  # position 4 sees tokens 1..4


if __name__ == "__main__":
    unittest.main()
