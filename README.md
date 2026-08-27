# Chest X-Ray Domain-Shift & Shortcut-Learning Audit

Pneumonia detection trained on a single-institution pediatric dataset and
stress-tested on two independent adult-population datasets — not just "how
accurate is it," but *why* it fails once it leaves the hospital it was
trained in, and whether that failure is a testable mechanism or a vague
"domain shift" excuse.

## Data

- **Training**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — pediatric, single institution, re-split at the **patient level**.
- **External validation only**: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) and [Indiana University/OpenI](https://openi.nlm.nih.gov/faq) — adult, two other institutions, never trained on.

None of these are committed to the repo — see [docs/data_setup.md](docs/data_setup.md) for setup (including why CheXpert/VinDr-CXR were ruled out).

## Status

Complete: data pipeline, three trained architectures (baseline CNN, ResNet-50, DenseNet-121), three-way domain-shift eval, Grad-CAM shortcut metric, causal analysis, Streamlit demo, 52 tests passing. Full findings in [docs/report.md](docs/report.md) and [docs/presentation.pptx](docs/presentation.pptx); day-by-day build log in [docs/roadmap.md](docs/roadmap.md); latest numbers in [docs/results.md](docs/results.md).

## Running it

```bash
pip install -r requirements.txt
bash scripts/download_kaggle.sh                                # 1. get the data (see docs/data_setup.md)
python scripts/train.py --model baseline_cnn --epochs 5         # 2. train
python scripts/evaluate_domain_shift.py --model baseline_cnn    # 3. domain-shift eval + CIs
python scripts/interpret.py --model baseline_cnn                # 4. Grad-CAM + shortcut metric
python scripts/causal_analysis.py --model baseline_cnn           # 5. causal analysis
python scripts/generate_results_table.py                        # 6. results table
streamlit run app/demo.py                                        # 7. inference demo
pytest                                                           # test suite
```

## Repo layout

```
src/data/       dataset loading, patient-level split, label harmonization
src/models/     baseline CNN, ResNet-50 / DenseNet-121 transfer learning
src/eval/       metrics, bootstrapped CIs, McNemar's test, domain-shift eval
src/interpret/  Grad-CAM, shortcut metric, failure taxonomy, causal analysis
app/            Streamlit inference demo
scripts/        training / evaluation / interpretation entry points
tests/          unit tests (pytest)
docs/           report, presentation, roadmap, data setup, results
```
