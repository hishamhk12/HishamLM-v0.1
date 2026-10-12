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
    val_fraction: float = 0.1  # legacy data_dir mode only
    dataset: str = ""  # prepared dataset dir (data.prepare); when set, data_dir/val_fraction are ignored
    sampling: str = "weights"  # "weights" or "weights_language_balanced" (from the dataset's meta.json)
    domain_share: float | None = None  # override the After-Sales sampling share (general sources rescaled)
    tokenizer: str = ""  # tokenizer JSON (e.g. a BPE file); empty = the dataset's character vocab.json

    # model (shared)
    context_length: int = 16
    embedding_dim: int = 24

    # model (mlp only)
    hidden_size: int = 128

    # model (transformer only)
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0
    tie_weights: bool = False  # share the token embedding matrix with the LM head

    # optimization
    batch_size: int = 32
    learning_rate: float = 3e-3
    max_steps: int = 2000
    grad_clip: float = 1.0  # max gradient norm; 0 disables clipping
    lr_schedule: str = "constant"  # "constant" or "cosine"
    warmup_steps: int = 0  # linear warmup from 0 to learning_rate
    min_lr: float = 0.0  # cosine schedule ends here
    lr_decay_steps: int = 0  # cosine reaches min_lr at this step and stays there; 0 = max_steps
    early_stopping_patience: int = 0  # stop after this many evaluations without a better val loss; 0 = off

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
