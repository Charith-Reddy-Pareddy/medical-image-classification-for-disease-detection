# Data setup

Datasets are never committed to git (too large, and most require accepting
a license/data-use agreement). This documents where each one goes.

## Kaggle Chest X-Ray Images (Pneumonia)

```bash
pip install kaggle
# requires ~/.kaggle/kaggle.json with your API credentials
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/ --unzip
```

Or download manually from
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia and
extract into `data/`. Either way you should end up with:

```
data/chest_xray/train/NORMAL/...
data/chest_xray/train/PNEUMONIA/...
data/chest_xray/test/NORMAL/...
data/chest_xray/test/PNEUMONIA/...
```

The Kaggle-provided train/test split is **not used as-is** — patient IDs are
parsed from filenames (e.g. `person1946_bacteria_4874.jpeg` -> patient
`1946`) and the dataset is re-split at the patient level in
`src/data/split.py`, since the official split places images from the same
patient in both train and test.

## NIH ChestX-ray14 (external validation only)

Download from https://nihcc.app.box.com/v/ChestXray-NIHCC and place under
`data/nih_chestxray14/`, keeping `images/` and the `Data_Entry_2017.csv`
metadata file. Never trained on — used only for out-of-domain evaluation.

## CheXpert (external validation only)

Requires accepting Stanford's data-use agreement:
https://stanfordmlgroup.github.io/competitions/chexpert/. Place under
`data/chexpert/`, keeping the image directories and `train.csv`/`valid.csv`.
Never trained on — used only for out-of-domain evaluation.
