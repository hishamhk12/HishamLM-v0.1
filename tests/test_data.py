import unittest

import torch

from training.data import get_batch, train_val_split


class TestData(unittest.TestCase):
    def test_split(self):
        ids = torch.arange(100)
        train, val = train_val_split(ids, 0.1)
        self.assertEqual(len(train), 90)
        self.assertEqual(len(val), 10)
        self.assertTrue(torch.equal(torch.cat([train, val]), ids))

    def test_batch_targets_are_shifted_inputs(self):
        data = torch.arange(50)
        gen = torch.Generator().manual_seed(0)
        x, y = get_batch(data, batch_size=4, context_length=8, generator=gen)
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))
        # data is 0..49, so the target of each token is that token + 1
        self.assertTrue(torch.equal(y, x + 1))

    def test_batches_are_reproducible(self):
        data = torch.arange(50)
        a = get_batch(data, 4, 8, torch.Generator().manual_seed(1))
        b = get_batch(data, 4, 8, torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(a[0], b[0]))

    def test_too_little_data(self):
        with self.assertRaises(ValueError):
            get_batch(torch.arange(8), 2, 8)


if __name__ == "__main__":
    unittest.main()
