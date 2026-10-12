"""A small decoder-only (GPT-style) Transformer language model with pre-norm blocks.

    token ids (B, T)
      -> token embedding + learned positional embedding   (B, T, C)
      -> n_layers x TransformerBlock                       (B, T, C)
      -> final LayerNorm
      -> linear LM head                                    (B, T, vocab_size)
"""

import math

import torch
import torch.nn as nn

from model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise MLP: C -> 4C -> C, applied to every position independently."""

    def __init__(self, embedding_dim, dropout=0.0):
        super().__init__()
        self.fc_in = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """Pre-norm block: normalize, transform, then add back to the residual stream."""

    def __init__(self, embedding_dim, n_heads, context_length, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, n_heads, context_length, dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # tokens exchange information
        x = x + self.ffn(self.ln2(x))  # each token is processed on its own
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim, n_heads, n_layers, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, n_heads, context_length, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        self._init_weights(n_layers)

    def _init_weights(self, n_layers):
        # Random init as in GPT-2: small normal weights, zero biases. LayerNorm
        # keeps its default (weight 1, bias 0).
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        # Layers that write into the residual stream get a smaller std, so the
        # stream's variance does not grow with the number of blocks.
        for block in self.blocks:
            nn.init.normal_(block.attn.W_o.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(block.ffn.fc_out.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, idx):
        B, T = idx.shape
        if T > self.context_length:
            raise ValueError(f"sequence length {T} exceeds context_length {self.context_length}")

        positions = torch.arange(T, device=idx.device)  # (T,)
        x = self.token_embedding(idx) + self.position_embedding(positions)  # (B, T, C)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
