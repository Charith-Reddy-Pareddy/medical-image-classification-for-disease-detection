# Data handling

This guide sets the starting rules for local dataset use. The planned sources and their roles are listed in the project README.

## Before downloading

For each source, check its current access terms and permitted use. Record the release or download date, source URL, label definition, and any exclusions needed for evaluation. Confirm whether patient or study identifiers are available before designing splits or uncertainty estimates.

Do not assume that datasets use compatible labels. Write down the mapping and ambiguous cases before combining or comparing labels.

## Local storage

Keep downloaded files outside version control under `data/`, separated by source. Planned paths are listed in [datasets.toml](../configs/datasets.toml); they are relative to the repository root. The manifest describes intended inputs, and the loaders are not wired to it yet.

Do not commit radiographs, archives, patient or study identifiers, dataset exports, or derived tables that could reveal records. Keep model checkpoints outside Git as well; they can be large and may contain embedded metadata.

## Credentials

Keep Kaggle credentials and other tokens outside the repository. Never add credential files, environment files, or copied secrets to a commit. If a credential is exposed, revoke it at its source.

## Source review

Reviewed 2026-11-05. Recheck source pages and terms when downloading; dataset pages and releases can change.

- **Kaggle Chest X-Ray Images (Pneumonia):** The listing describes Normal and Pneumonia classes. Its description and linked Mendeley record identify CC BY 4.0. Record the exact download version and file checksums; do not assume counts or file layouts are identical across snapshots. [Kaggle listing](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) · [Mendeley record, version 2](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- **NIH ChestX-ray14:** NIH's official data page says there are no use restrictions, and asks users to link the NIH download site, cite the CVPR 2017 paper, and acknowledge the NIH Clinical Center. The Box portal itself was not accessible during this review, so record the exact downloaded release and recheck its supplied documentation. A local legacy CSV header contains `Finding Labels` and `Patient ID`; verify these fields against the rebuild's download. The labels are mined from reports and may include multiple findings per image. [NIH data guidance](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) · [NIH download portal](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [Original paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)
- **Indiana University/OpenI:** The NLM FAQ describes 7,470 chest X-rays and 3,955 reports, with image-report associations available in XML. NLM says reuse follows the applicable image license; the Indiana University images are distributed under CC BY-NC-ND 4.0, and the FAQ asks that the dataset not be shared outside the research group or organization. It also requests NLM attribution and citation. Validate the report-to-image links and the Pneumonia label mapping before evaluation. [OpenI FAQ](https://openi-vip.nlm.nih.gov/faq)

No cross-dataset binary label mapping has been established yet. Keep each source's labels distinct until the mapping and ambiguous cases have been reviewed.

## Grouping and split units

A local review of the earlier project materials found usable grouping keys, with limits that must be checked again against the exact download used by this rebuild:

- **Kaggle:** In the local 5,856-image snapshot, the filename parser recognized every name and found 170 patient-key overlaps between the supplied train and test folders. Use filename-derived patient keys for grouping, and verify parser coverage and split overlap for every new download before using its split.
- **NIH:** The local metadata CSV includes a `Patient ID` column. Use that field as the grouping key for patient-level splits and clustered evaluation when NIH is used.
- **Indiana University/OpenI:** The XML links images to report records, and some reports link multiple images. A report identifier can group images from the same report, but it has not been verified as a person-level identifier across reports. Describe any resampling by report/study unless a stable patient key is confirmed.

The local counts above describe an earlier snapshot only. They do not establish the structure or split quality of a future download.

## Reproducibility

Keep non-sensitive source details and label-mapping decisions in the project. Before reporting results, verify the data version and split logic used by the run. Do not publish results from data whose access terms or provenance are unclear.

## Earlier project materials

The earlier implementation, saved checkpoints, and reported results are not included in this repository. Treat them as reference material until their provenance and applicable dataset terms have been reviewed. Confirm the data and configuration behind an artifact before reuse, and reproduce reported metrics with the rebuilt pipeline.
