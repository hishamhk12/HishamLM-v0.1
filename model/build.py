from model.mlp_lm import MLPLanguageModel
from model.transformer_lm import TransformerLanguageModel

MODEL_TYPES = ("mlp", "transformer")


def build_model(cfg, vocab_size):
    """Create a randomly initialized model of type `cfg.model_type`."""
    if cfg.model_type == "mlp":
        return MLPLanguageModel(
            vocab_size=vocab_size,
            context_length=cfg.context_length,
            embedding_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_size,
        )
    if cfg.model_type == "transformer":
        return TransformerLanguageModel(
            vocab_size=vocab_size,
            context_length=cfg.context_length,
            embedding_dim=cfg.embedding_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            tie_weights=cfg.tie_weights,
        )
    raise ValueError(f"unknown model_type {cfg.model_type!r}, expected one of {MODEL_TYPES}")
