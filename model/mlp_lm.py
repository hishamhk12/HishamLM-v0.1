"""A small MLP language model (in the spirit of Bengio et al., 2003).

For every position t the model looks at a fixed window of the last
`context_length` tokens (t - n + 1 ... t), concatenates their embeddings,
passes them through one hidden tanh layer and predicts token t + 1.

Input:  token ids        (B, T)
Output: next-token logits (B, T, vocab_size)

This is the same input/output contract a Transformer will use later, so the
data pipeline, training loop and generation code can be reused as-is.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden = nn.Linear(context_length * embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

        # Shrink the output layer so the initial predictions are close to
        # uniform, giving an initial loss near ln(vocab_size).
        with torch.no_grad():
            self.output.weight.mul_(0.1)
            self.output.bias.zero_()

    def forward(self, idx):
        B, T = idx.shape
        n = self.context_length

        emb = self.token_embedding(idx)  # (B, T, C)

        # Left-pad with n - 1 zero vectors so the first positions also have a
        # full window. Position t in the original sequence sits at t + n - 1.
        padded = F.pad(emb, (0, 0, n - 1, 0))  # (B, T + n - 1, C)

        # Slice i gives, for every position t, the token at t - (n - 1) + i.
        # Concatenating all n slices builds each position's window; it only
        # contains tokens at or before t, so the model cannot see the future.
        windows = torch.cat([padded[:, i : i + T, :] for i in range(n)], dim=-1)  # (B, T, n*C)

        h = torch.tanh(self.hidden(windows))  # (B, T, hidden)
        return self.output(h)  # (B, T, vocab_size)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
