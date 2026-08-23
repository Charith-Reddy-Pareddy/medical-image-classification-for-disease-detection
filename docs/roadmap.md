# Build roadmap

Built incrementally over about a week. Checked off as each day lands.

- [x] **Day 1** — project scaffold, environment, Kaggle download instructions, README
- [x] **Day 2** — patient-level train/val/test split, label harmonization, Dataset/DataLoader
- [x] **Day 3** — baseline CNN (from scratch), training loop, metrics
- [x] **Day 4** — ResNet-50 and DenseNet-121 transfer learning
- [x] **Day 5** — bootstrapped CIs, McNemar's test, domain-shift eval harness (Kaggle -> NIH -> OpenI)
- [x] **Day 6** — Grad-CAM, quantitative shortcut-feature metric, error taxonomy
- [x] **Day 7** — added regression tests for the split/label-harmonization edge cases
- [x] **Day 8** — age-artifact causal analysis, Streamlit demo, results table, polish

That's the full 8-day build, since extended: all three architectures are
now properly trained and the three-way domain-shift comparison
(Kaggle -> NIH -> OpenI) is complete. See [docs/results.md](results.md)
for current numbers and the README's "Status" section for what's
genuinely finished.

See the [README](../README.md) for the research questions this is answering.
