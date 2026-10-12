import json
from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    run_name: str = "tiny_char"
    data_dir: str = "data/raw"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu" or "cuda"

    # data
    val_fraction: float = 0.1

    # model
    context_length: int = 16
    embedding_size: int = 24
    hidden_size: int = 128

    # optimization
    batch_size: int = 32
    learning_rate: float = 3e-3
    max_steps: int = 2000

    # evaluation / logging
    eval_interval: int = 200
    eval_batches: int = 20

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))  # unknown keys raise a TypeError

    def to_dict(self):
        return asdict(self)
