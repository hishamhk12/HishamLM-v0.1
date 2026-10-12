import random

import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
