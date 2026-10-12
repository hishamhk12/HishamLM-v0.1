"""Causal multi-head self-attention, written out step by step.

Shapes used below:
    B = batch size, T = sequence length, C = embedding_dim,
    H = n_heads,    D = head_dim = C / H
"""

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, context_length, dropout=0.0):
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads}); "
                f"each head gets embedding_dim / n_heads dimensions"
            )
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        # One projection per role. Each produces all H heads at once (C = H * D).
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Mixes the concatenated heads back into one C-dimensional vector.
        self.W_o = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular matrix: mask[i, j] is True when position i may look at j (j <= i).
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        # 1. Queries, keys and values: Q = X Wq, K = X Wk, V = X Wv
        Q = self.W_q(x)  # (B, T, C)
        K = self.W_k(x)  # (B, T, C)
        V = self.W_v(x)  # (B, T, C)

        # 2. Split the C dimensions into H heads of D dimensions each, and move
        #    the head axis before T so every head is an independent attention.
        Q = Q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        V = V.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        # 3. Scores: how well each query matches each key, scaled by 1/sqrt(D)
        #    so their variance does not grow with the head size.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, T)

        # 4. Causal mask: position i must not see positions j > i. Setting those
        #    scores to -inf makes their softmax weight exactly 0.
        mask = self.causal_mask[:T, :T]  # (T, T)
        scores = scores.masked_fill(~mask, float("-inf"))

        # 5. Each row becomes a probability distribution over the visible positions.
        weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        weights = self.attn_dropout(weights)

        # 6. Weighted sum of the values.
        out = torch.matmul(weights, V)  # (B, H, T, D)

        # 7. Merge the heads back: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output projection.
        out = self.resid_dropout(self.W_o(out))  # (B, T, C)

        if return_weights:
            return out, weights
        return out
