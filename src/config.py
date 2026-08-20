import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))

KAGGLE_DIR = DATA_DIR / "chest_xray"
NIH_DIR = DATA_DIR / "nih_chestxray14"
CHEXPERT_DIR = DATA_DIR / "chexpert"

IMAGE_SIZE = 224
SEED = 42
