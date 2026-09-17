# Chest X-Ray Domain-Shift & Shortcut-Learning Audit

[![tests](https://github.com/Charith-Reddy-Pareddy/medical-image-classification-for-disease-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Charith-Reddy-Pareddy/medical-image-classification-for-disease-detection/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**[Live results dashboard →](https://charith-reddy-pareddy.github.io/medical-image-classification-for-disease-detection/)**

Pneumonia detection trained on a single-institution pediatric dataset and
stress-tested on two independent adult-population datasets — not just "how
accurate is it," but *why* it fails once it leaves the hospital it was
trained in, and whether that failure is a testable mechanism or a vague
"domain shift" excuse.

```
Kaggle pediatric chest X-rays
        │
        ▼
Patient-level train/val/test split  (no leakage across sets)
        │
        ▼
Baseline CNN  /  ResNet-50  /  DenseNet-121   (trained once, on Kaggle only)
        │
        ├──────────────┬──────────────────────┐
        ▼              ▼                      ▼
   In-domain      External: NIH          External: OpenI
   (Kaggle test)  ChestX-ray14           Indiana Univ.
        │              │                      │
        └──────────────┴──────────────────────┘
                        ▼
        Metrics + bootstrapped CIs + calibration
                        ▼
              Grad-CAM attention maps
                        ▼
     Overlap vs. reference lung segmentation
                        ▼
   Border-masking intervention + threshold sensitivity
```

## Key findings

- **ResNet-50 hits 0.992 AUC-ROC in-domain**, but external AUC-ROC drops
  to **0.707 (NIH)** and **0.575 (OpenI)** — and the baseline CNN's OpenI
  AUC-ROC (**0.512**, 95% CI 0.469–0.556) is statistically indistinguishable
  from chance.
- **Patient-level splitting** (not just image-level) prevents the same
  patient's scans from leaking across train/val/test.
- **74–81% of *correctly classified* predictions**, across all three
  architectures, show Grad-CAM attention concentrated outside a reference
  lung segmentation — shortcut reliance is the median behavior, not a tail
  risk, and it holds across a Grad-CAM-threshold/overlap-cutoff sensitivity
  sweep, not just one cutoff (`scripts/shortcut_sensitivity.py`).
- **A border-masking intervention corroborates the Grad-CAM finding**:
  blanking the outer 10% of each image costs the from-scratch baseline CNN
  14.7 accuracy points but only 1.3–3.3 for the transfer models — an
  independent, interventional check, not just a post-hoc visualization.
- **A multivariable association analysis** (explicitly not a causal claim)
  finds shortcut reliance tracks image resolution and detected scanner
  artifacts more consistently than a raw pediatric-vs-adult age label.

## Experiments

| # | Question | Method | Finding |
|---|---|---|---|
| E1 | Can a CNN classify pneumonia at all? | Baseline CNN, from scratch | 0.975 AUC-ROC in-domain |
| E2 | Does transfer learning help? | ResNet-50 / DenseNet-121 vs. baseline | ≈0.99 AUC-ROC, McNemar's p<0.001 |
| E3 | Does it generalize to other hospitals? | Eval on NIH / OpenI, no retraining | Large, architecture-dependent degradation |
| E4 | Is the model calibrated, or just discriminative? | Brier score, ECE, AUPRC at low prevalence | See [docs/results.md](docs/results.md) |
| E5 | Where does the model actually look? | Grad-CAM vs. reference lung segmentation | 74–81% low-overlap on correct predictions |
| E6 | Does removing border artifacts change predictions? | Border-masking intervention | Architecture-dependent accuracy drop |
| E7 | Does the shortcut-metric conclusion depend on its thresholds? | cam_threshold × overlap_cutoff sweep | Conclusion holds across the swept range |
| E8 | What predicts shortcut reliance? | Multivariable logistic regression | Resolution/artifact proxies, not a clean age effect |

## Results

Pneumonia vs. normal, binary classification. In-domain (Kaggle, held-out
patient-level test set, n=880):

| Model | Accuracy | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| Baseline CNN (from scratch) | 93.6% | 96.8% | 94.0% | 0.975 |
| ResNet-50 (transfer) | **97.2%** | 97.0% | 99.0% | **0.992** |
| DenseNet-121 (transfer) | 96.4% | 96.5% | 98.4% | 0.991 |

That accuracy does not survive leaving the hospital it was trained in —
this is the actual point of the project:

![Domain shift AUC-ROC by architecture and site](docs/report_assets/hero_domain_shift_auc.png)

Baseline CNN's AUC-ROC on the OpenI external site is 0.512 (95% CI
0.469–0.556) — statistically indistinguishable from a coin flip. Full
numbers with confidence intervals: [docs/results.md](docs/results.md).
Full writeup, including *why* it fails (a quantitative Grad-CAM audit finds
Grad-CAM attention falls outside a reference lung mask in 74–81% of even
*correct* predictions):
[docs/report.md](docs/report.md).

## Research questions

- How accurately do CNN architectures detect pneumonia, and what drives misclassification?
- How much does performance degrade across two independent external datasets?
- Do a custom CNN, ResNet-50, and DenseNet-121 differ meaningfully in generalization?
- Is the model attending to lung tissue, or exploiting shortcut features — measured quantitatively, not eyeballed?
- Does shortcut reliance track pediatric-vs-adult imaging protocol, or a more directly measurable proxy?

## Data

- **Training**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — pediatric, single institution, re-split at the **patient level**.
- **External validation only**: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) and [Indiana University/OpenI](https://openi.nlm.nih.gov/faq) — adult, two other institutions, never trained on.

None of these are committed to the repo — see [docs/data_setup.md](docs/data_setup.md) for setup (including why CheXpert/VinDr-CXR were ruled out).

## Status

Complete: data pipeline, three trained architectures (baseline CNN, ResNet-50, DenseNet-121), three-way domain-shift eval, Grad-CAM shortcut metric, age-artifact association analysis, Streamlit demo, 63 tests passing. Full findings in [docs/report.md](docs/report.md) and [docs/presentation.pptx](docs/presentation.pptx); day-by-day build log in [docs/roadmap.md](docs/roadmap.md); latest numbers in [docs/results.md](docs/results.md).

## Running it

```bash
pip install -r requirements.txt
bash scripts/download_kaggle.sh                                # 1. get the data (see docs/data_setup.md)
python scripts/train.py --model baseline_cnn --epochs 5         # 2. train
python scripts/evaluate_domain_shift.py --model baseline_cnn    # 3. domain-shift eval + CIs
python scripts/interpret.py --model baseline_cnn                # 4. Grad-CAM + shortcut metric
python scripts/mechanism_analysis.py --model baseline_cnn        # 5. age-artifact association analysis
python scripts/shortcut_sensitivity.py --model baseline_cnn      # 6. threshold sensitivity sweep
python scripts/generate_results_table.py                        # 7. results table
streamlit run app/demo.py                                        # 8. inference demo
pytest                                                           # test suite
```

## Repo layout

```
src/data/       dataset loading, patient-level split, label harmonization
src/models/     baseline CNN, ResNet-50 / DenseNet-121 transfer learning
src/eval/       metrics, bootstrapped CIs, McNemar's test, domain-shift eval
src/interpret/  Grad-CAM, shortcut metric, failure taxonomy, association analysis
app/            Streamlit inference demo
scripts/        training / evaluation / interpretation entry points
tests/          unit tests (pytest)
docs/           report, presentation, roadmap, data setup, results
```
