# Build roadmap

Built incrementally over about a week. Checked off as each day lands.

- [x] **Day 1** — project scaffold, environment, Kaggle download instructions, README
- [x] **Day 2** — patient-level train/val/test split, label harmonization, Dataset/DataLoader
- [x] **Day 3** — baseline CNN (from scratch), training loop, metrics
- [x] **Day 4** — ResNet-50 and DenseNet-121 transfer learning
- [x] **Day 5** — bootstrapped CIs, McNemar's test, domain-shift eval harness (Kaggle -> NIH; VinDr-CXR pending download)
- [x] **Day 6** — Grad-CAM, quantitative shortcut-feature metric, error taxonomy
- [x] **Day 7** — debugging exercise (intentional defects, diagnosis, fixes, regression tests) — see [docs/debugging_exercise.md](debugging_exercise.md)
- [x] **Day 8** — age-artifact causal analysis, Streamlit demo, results table, polish

That's the full 8-day build. See [docs/results.md](results.md) for current
numbers and the README's "Status" section for what's genuinely finished
vs. what still needs a full training run (VinDr-CXR download, converged
checkpoints for all three architectures).

See the [README](../README.md) for the research questions this is answering.
