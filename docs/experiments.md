# Experiments and evaluation

## Data preparation

The project uses the flow-based anomaly-detection part of CIC IoT-DIAD 2024.
One row represents one network flow. Labels are inferred from the dataset's
directory structure: benign traffic is the negative class, and attack traffic
is the positive class.

The working split is recorded in
[`python_sample_summary.csv`](../results/python_sample_summary.csv):

- 30,000 benign flows for unsupervised training;
- 12,000 different benign flows for evaluation;
- 14,000 attack flows for evaluation, balanced across seven categories.

Sampling uses seeded hashes of record content and provenance. Training records
are explicitly excluded from evaluation using `(source_file, source_row)` keys.
This ensures record-level separation; it does not guarantee separation by
device, capture session or duplicated content in different source files.

The preprocessing excludes labels, source identifiers, IP addresses, timestamps
and flow IDs. Constant features are removed using training data. Median
imputation and standardisation are fitted on training data and then applied to
evaluation data. See [`data_utils.py`](../python/data_utils.py) and
[`python_preprocessing_summary.csv`](../results/python_preprocessing_summary.csv).

## Models

Clustering models learn a representation of benign traffic. Evaluation flows
are scored using distance to representatives, likelihood or density-core
distance, depending on the model. Thresholds turn these scores into anomaly
predictions. DBSCAN and OPTICS use the project's out-of-sample core-distance
rule rather than a native `predict` method.

Additional novelty detectors include a teaching-oriented custom Isolation
Forest, One-Class SVM, PCA reconstruction error and ECOD. HDBSCAN is used for
exploratory structure analysis, not as the main novelty detector.

Supervised Random Forest and HistGradientBoosting experiments use their own
mixed training/validation split and a shared test sample. They are not trained
under the same information constraints as the benign-only models.

## Selected operating points

These values are copied from the original recorded CSVs. The refined rows are
selected by maximum F1 within each model's sweep; the baseline rows use the
named configurations below.

| Model | Configuration | Precision | Recall | F1 | False-positive rate |
|---|---|---:|---:|---:|---:|
| k-means | `kmeans_max_f1` | 79.32% | 56.57% | 0.6604 | 17.21% |
| BIRCH | `birch_refined_t0.75_p91` | 81.10% | 54.42% | 0.6513 | 14.80% |
| DBSCAN | `dbscan_refined_eps0.26_m12` | 77.10% | 60.25% | 0.6764 | 20.88% |
| GMM | `gmm_refined_k32_tied_p10` | 78.35% | 30.86% | 0.4428 | 9.95% |
| Custom Isolation Forest | `iforest_higher_recall` | 66.75% | 20.65% | 0.3154 | 12.00% |

Sources: [`python_experiments.csv`](../results/python_experiments.csv) and
[`python_clustering_refined_summary.csv`](../results/python_clustering_refined_summary.csv).

False-positive rate is `FP / (FP + TN)`: the fraction of benign evaluation flows
flagged as attacks. It is not the fraction of alerts that are false. Precision
depends on the attack prevalence in this deliberately balanced evaluation set.

## Additional checks

- [`python_stability_summary.csv`](../results/python_stability_summary.csv):
  repeated runs with fixed seeds.
- [`python_clustering_train_scale_summary.csv`](../results/python_clustering_train_scale_summary.csv):
  benign training sizes of 30k, 50k and 100k with a fixed evaluation sample.
- [`python_unknown_attack_aggregate.csv`](../results/python_unknown_attack_aggregate.csv):
  leave-one-attack-class-out comparison with supervised models.
- [`python_clustering_refined_recall_by_class.csv`](../results/python_clustering_refined_recall_by_class.csv):
  recall differences between attack categories.

The benign-only models do not train on any attack class. The leave-one-class-out
condition therefore changes the supervised training data, not their own
training data. Their unchanged predictions should not be interpreted as an
additional demonstration of robustness.

## Limits of the experiment

The results are exploratory: several sweeps and operating-point selections use
the same evaluation set. A separate untouched test set is needed for a final
generalisation estimate. Record-level sampling also permits source/session
dependence, and the dataset's class balance differs from a production network.

This repository does not implement live packet capture, real-time alerts,
deployment or incident response. Runtime measurements depend on hardware and
are not end-to-end latency measurements. The Julia checkpoint covers an
earlier subset of the experiments rather than all Python stages.
