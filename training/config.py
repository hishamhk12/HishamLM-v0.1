import json
from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    run_name: str = "mlp_char"
    model_type: str = "mlp"  # "mlp" or "transformer"
    data_dir: str = "data/raw"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu" or "cuda"

    # data
    val_fraction: float = 0.1

    # model (shared)
    context_length: int = 16
    embedding_dim: int = 24

    # model (mlp only)
    hidden_size: int = 128

    # model (transformer only)
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0

    # optimization
    batch_size: int = 32
    learning_rate: float = 3e-3
    max_steps: int = 2000
    grad_clip: float = 1.0  # max gradient norm; 0 disables clipping

    # evaluation / logging
    eval_interval: int = 200
    eval_batches: int = 20

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "embedding_size" in d:  # phase-1 name for embedding_dim
            d["embedding_dim"] = d.pop("embedding_size")
        return cls(**d)  # unknown keys raise a TypeError

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self):
        return asdict(self)
