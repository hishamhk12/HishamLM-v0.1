import unittest

import torch

from inference.generate import generate
from model.attention import CausalSelfAttention
from model.transformer_lm import TransformerLanguageModel

V, T_MAX, C, H, L = 12, 8, 16, 4, 2


def small_model(dropout=0.0):
    return TransformerLanguageModel(V, context_length=T_MAX, embedding_dim=C, n_heads=H, n_layers=L, dropout=dropout)


class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attn = CausalSelfAttention(C, H, context_length=T_MAX)

    def test_head_dimensions(self):
        self.assertEqual(self.attn.head_dim, C // H)
        out, weights = self.attn(torch.randn(2, 5, C), return_weights=True)
        self.assertEqual(out.shape, (2, 5, C))
        self.assertEqual(weights.shape, (2, H, 5, 5))  # one T x T map per head

    def test_weights_are_causal(self):
        _, weights = self.attn(torch.randn(2, T_MAX, C), return_weights=True)
        future = torch.triu(torch.ones(T_MAX, T_MAX, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.all(weights[..., future] == 0))  # no weight on any j > i
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, H, T_MAX)))  # rows are distributions
        self.assertTrue(torch.allclose(weights[..., 0, 0], torch.ones(2, H)))  # first token sees only itself

    def test_invalid_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            CausalSelfAttention(embedding_dim=10, n_heads=4, context_length=T_MAX)


class TestTransformerLanguageModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = small_model()

    def test_output_shape(self):
        idx = torch.randint(V, (3, T_MAX))
        self.assertEqual(self.model(idx).shape, (3, T_MAX, V))
        self.assertEqual(self.model(idx[:, :3]).shape, (3, 3, V))

    def test_too_long_sequence(self):
        with self.assertRaises(ValueError):
            self.model(torch.randint(V, (1, T_MAX + 1)))

    def test_backward(self):
        idx = torch.randint(V, (2, T_MAX))
        loss = self.model(idx).logsumexp(-1).mean()
        loss.backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_future_tokens_do_not_change_past_logits(self):
        idx = torch.randint(V, (1, T_MAX))
        for t in range(T_MAX):
            changed = idx.clone()
            changed[0, t:] = (idx[0, t:] + 1) % V  # change token t and everything after it
            a, b = self.model(idx), self.model(changed)
            self.assertTrue(torch.allclose(a[:, :t], b[:, :t], atol=1e-6), f"position < {t} saw the future")
            self.assertFalse(torch.allclose(a[:, t], b[:, t]))  # but position t itself does change

    def test_initial_loss_is_near_uniform(self):
        idx = torch.randint(V, (4, T_MAX))
        logits = self.model(idx)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, V), idx.view(-1))
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(float(V))).item(), delta=0.1)

    def test_generation(self):
        self.model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        out = generate(self.model, prompt, max_new_tokens=20, temperature=1.0, top_k=5,
                       generator=torch.Generator().manual_seed(0))
        self.assertEqual(out.shape, (1, 23))  # longer than context_length: the context is cropped
        self.assertTrue(torch.equal(out[:, :3], prompt))
        self.assertTrue(((out >= 0) & (out < V)).all())

    def test_greedy_generation_is_deterministic(self):
        self.model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        a = generate(self.model, prompt, 10, temperature=0)
        b = generate(self.model, prompt, 10, temperature=0)
        self.assertTrue(torch.equal(a, b))
        expected = self.model(a[:, -T_MAX - 1 : -1])[:, -1].argmax(-1)  # the last token was the argmax
        self.assertEqual(a[0, -1].item(), expected.item())


if __name__ == "__main__":
    unittest.main()


class TestWeightTying(unittest.TestCase):
    def test_tied_head_shares_the_embedding(self):
        import tempfile
        from pathlib import Path

        untied = TransformerLanguageModel(V, T_MAX, C, H, L)
        tied = TransformerLanguageModel(V, T_MAX, C, H, L, tie_weights=True)
        self.assertIs(tied.lm_head.weight, tied.token_embedding.weight)
        self.assertEqual(untied.num_parameters() - tied.num_parameters(), V * C)
        for model in (tied, untied):
            parts = model.parameter_breakdown()
            self.assertEqual(parts["total"], model.num_parameters())
        self.assertEqual(tied.parameter_breakdown()["lm_head"], 0)

        # gradients from both uses reach the shared matrix, and saving/loading keeps the tie
        tied(torch.randint(V, (2, T_MAX))).sum().backward()
        self.assertIsNotNone(tied.token_embedding.weight.grad)
        with tempfile.TemporaryDirectory() as d:
            torch.save(tied.state_dict(), Path(d) / "m.pt")
            loaded = TransformerLanguageModel(V, T_MAX, C, H, L, tie_weights=True)
            loaded.load_state_dict(torch.load(Path(d) / "m.pt"))
        self.assertIs(loaded.lm_head.weight, loaded.token_embedding.weight)
        self.assertTrue(torch.equal(loaded.lm_head.weight, tied.token_embedding.weight))
