# Chest X-Ray Domain-Shift & Shortcut-Learning Audit

Pneumonia detection on chest X-rays, trained on a single-institution pediatric
dataset and stress-tested on two independent adult-population datasets. The
goal isn't just "build a classifier" — it's to measure *why* medical imaging
models fail when they leave the hospital they were trained in, and whether
that failure is explained by a specific, testable mechanism (imaging
protocol/scanner differences) rather than a vague "domain shift" excuse.

## Research questions

- How accurately can CNN-based architectures detect pneumonia, and what
  drives misclassification?
- How much does performance degrade across two independent external
  datasets (NIH ChestX-ray14, VinDr-CXR)?
- Do a custom CNN, ResNet-50, and DenseNet-121 differ meaningfully in
  sensitivity, specificity, and parameter efficiency?
- Is the model looking at lung tissue, or at shortcut features (borders,
  text markers, resolution artifacts) — measured quantitatively, not
  eyeballed from a handful of images?
- Does shortcut reliance track pediatric-vs-adult imaging protocol
  specifically, via a causal (logistic regression) analysis rather than an
  unexplained correlation?

## Data

- **Training**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
  — pediatric, single institution. Re-split at the **patient level** (parsed
  from filenames) before training, since the official split leaks patients
  across train/test.
- **External validation only, never trained on**:
  [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) and
  [VinDr-CXR](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)
  (two Vietnamese hospitals). Both adult population, different institutions.
  CheXpert was the original third site, but Stanford has since moved it
  behind an institutional access agreement with no fixed approval
  timeline — see [docs/data_setup.md](docs/data_setup.md) for details and
  why VinDr-CXR replaces it.

None of these datasets are committed to this repo — see
[docs/data_setup.md](docs/data_setup.md) for how to fetch them.

## Status

Work in progress, built incrementally over a ~week-long build. See
[docs/roadmap.md](docs/roadmap.md) for the day-by-day plan and current
progress.

## Repo layout

```
src/data/       dataset loading, patient-level split, label harmonization
src/models/     baseline CNN, ResNet-50 / DenseNet-121 transfer learning
src/eval/       metrics, bootstrapped CIs, McNemar's test, domain-shift eval
src/interpret/  Grad-CAM, shortcut-feature metric, causal (age) analysis
app/            Streamlit inference demo
scripts/        data download / prep entry points
tests/          unit tests
docs/           roadmap, data setup, report
```
