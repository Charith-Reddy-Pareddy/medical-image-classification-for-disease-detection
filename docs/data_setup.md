# Data setup

Datasets are never committed to git (too large, and most require accepting
a license/data-use agreement). This documents where each one goes.

## Kaggle Chest X-Ray Images (Pneumonia) — downloaded, verified

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

## NIH ChestX-ray14 (external validation only) — downloaded, verified

Download from https://nihcc.app.box.com/v/ChestXray-NIHCC and place under
`data/nih_chestxray14/`, keeping `images/` (112,120 PNGs) and the
`Data_Entry_2017_v2020.csv` metadata file (has the `Finding Labels` column
`harmonize_nih()` expects). Never trained on — used only for out-of-domain
evaluation.

Note: the public Box folder has had spam PDFs ("AAA Job Opportunity...")
uploaded alongside the real files by unrelated third parties — exclude
anything not matching the official file list (images archives, CSVs,
`README_CHESTXRAY.pdf`, `FAQ_CHESTXRAY.pdf`, `ARXIV_V5_CHESTXRAY.pdf`,
`LOG_CHESTXRAY.pdf`, `batch_download_zips.py`) when extracting.

## VinDr-CXR (external validation only, replaces CheXpert) — not yet downloaded

The proposal originally called for CheXpert as the third site, but Stanford
has since moved access behind an AIMI membership + signed Research
Agreement (both requiring manual institutional review, no fixed timeline)
— see https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x. VinDr-CXR is
used instead: 18,000 adult chest X-rays from two Vietnamese hospitals
(Hospital 108, Hanoi Medical University Hospital), radiologist-labeled
across 28 findings including Pneumonia — published in *Scientific Data*
(https://www.nature.com/articles/s41597-022-01498-w). It still gives a
third, independent institution for the domain-shift comparison, adult
population same as the original CheXpert plan.

```bash
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection -p data/
```

(Requires accepting the competition rules on the Kaggle site once, then
the CLI download is immediate — no application/approval wait, unlike the
official PhysioNet release of the same data, which is credentialed.)
Place under `data/vindr_cxr/`, keeping the image directory and
`train.csv` (or equivalent findings CSV). Never trained on — used only
for out-of-domain evaluation.

If CheXpert access is later approved, its harmonization can still be
added back in `src/data/labels.py` alongside VinDr-CXR's.
