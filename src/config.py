import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))

KAGGLE_DIR = DATA_DIR / "chest_xray"
NIH_DIR = DATA_DIR / "nih_chestxray14"
CHEXPERT_DIR = DATA_DIR / "chexpert"

IMAGE_SIZE = 224
SEED = 42
