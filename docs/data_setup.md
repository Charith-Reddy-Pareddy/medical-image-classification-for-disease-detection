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

## Indiana University / OpenI (external validation only, replaces CheXpert) — downloaded, verified

The proposal originally called for CheXpert as the third site; Stanford has
since moved access behind an AIMI membership + signed Research Agreement
(manual institutional review, no fixed timeline) — see
https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x. VinDr-CXR (Kaggle's
`vinbigdata-chest-xray-abnormalities-detection` competition) was tried next,
but ruled out on inspection: the competition repackaging is 142GB of raw
DICOM (didn't fit available disk space) and, more fundamentally, its
14-class taxonomy doesn't include a "Pneumonia" label at all (only the
*original* VinDr-CXR research release's 28-label version does — a citation
mismatch on my part when first proposing it).

Indiana University Health's OpenI collection is used instead: 7,470 chest
X-rays with radiology reports, MeSH-tagged, explicitly including a
"Pneumonia" label — verified directly against `torchxrayvision`'s loader
source and the raw XML reports before committing to it, not assumed from a
paper abstract this time. Public, no access agreement, ~1.3GB total.

```bash
curl -L "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz" -o data/openi/NLMCXR_png.tgz
curl -L "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz" -o data/openi/NLMCXR_reports.tgz
mkdir -p data/openi/images data/openi/reports
tar -xzf data/openi/NLMCXR_png.tgz -C data/openi/images/
tar -xzf data/openi/NLMCXR_reports.tgz -C data/openi/reports/
```

Reports are per-patient XML files under `data/openi/reports/ecgen-radiology/`,
each MeSH-tagged with curated ("major") and NLP-extracted ("automatic")
terms; `src/data/openi.py` parses these and `harmonize_openi()` in
`src/data/labels.py` applies label=1 if "pneumonia" is among the automatic
tags, label=0 if the major tags are exactly "normal", excluding ambiguous
cases otherwise — same policy as `harmonize_nih()`. 2,854 usable images
after harmonization (2,696 normal / 158 pneumonia). Never trained on — used
only for out-of-domain evaluation.

If CheXpert access is later approved, its harmonization can still be added
back in `src/data/labels.py` alongside OpenI's.
