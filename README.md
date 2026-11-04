# Chest X-Ray Domain Shift and Shortcut Learning

This project studies how chest X-ray classifiers behave when evaluated beyond the data source used for training, and whether their predictions may rely on image features outside the lungs. It is a research project, not a clinical diagnostic tool.

## Questions

- How well do models trained on one chest X-ray source perform on held-out and external data?
- How do a small CNN, ResNet-50, and DenseNet-121 compare under the same evaluation setup?
- How often does Grad-CAM localization fall outside an estimated lung region, and how sensitive is that measure to its thresholds?
- Do masking interventions change model predictions or performance?
- How are shortcut indicators associated with measurable image characteristics?

## Data plan

The planned training source is the Kaggle Chest X-Ray Images (Pneumonia) dataset. NIH ChestX-ray14 and Indiana University/OpenI are planned for external evaluation only. Before using any dataset, the project will verify access terms, data versions, labels, and patient or study identifiers. Data and credentials will not be committed to this repository.

## Interpretation limits

A regression over observed image characteristics can describe associations; it does not by itself identify causal effects. Grad-CAM overlap with a model-produced lung mask is a localization measure, not proof of what the classifier relies on. Comparisons between datasets can reflect differences in population, prevalence, acquisition, labeling, and institution.

## Development setup

Use Python 3.10–3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Status

This repository is being rebuilt. Previously reported metrics and conclusions have not yet been reproduced in this repository and should not be treated as verified results.
