#!/usr/bin/env bash
# Downloads the Kaggle Chest X-Ray Images (Pneumonia) dataset into data/.
# Requires `pip install kaggle` and ~/.kaggle/kaggle.json with your API token.
set -euo pipefail

cd "$(dirname "$0")/.."
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/ --unzip
echo "Done. Expect data/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}."
