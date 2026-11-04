# Data handling

This guide sets the starting rules for local dataset use. The planned sources and their roles are listed in the project README.

## Before downloading

For each source, check its current access terms and permitted use. Record the release or download date, source URL, label definition, and any exclusions needed for evaluation. Confirm whether patient or study identifiers are available before designing splits or uncertainty estimates.

Do not assume that datasets use compatible labels. Write down the mapping and ambiguous cases before combining or comparing labels.

## Local storage

Keep downloaded files outside version control under `data/`, separated by source:

```text
data/
  kaggle/
  nih/
  openi/
```

Do not commit radiographs, archives, patient or study identifiers, dataset exports, or derived tables that could reveal records. Keep model checkpoints outside Git as well; they can be large and may contain embedded metadata.

## Credentials

Keep Kaggle credentials and other tokens outside the repository. Never add credential files, environment files, or copied secrets to a commit. If a credential is exposed, revoke it at its source.

## Reproducibility

Keep non-sensitive source details and label-mapping decisions in the project. Before reporting results, verify the data version and split logic used by the run. Do not publish results from data whose access terms or provenance are unclear.

## Earlier project materials

The earlier implementation, saved checkpoints, and reported results are not included in this repository. Treat them as reference material until their provenance and applicable dataset terms have been reviewed. Confirm the data and configuration behind an artifact before reuse, and reproduce reported metrics with the rebuilt pipeline.
