# Python experiment guide

Run scripts from the repository root. The names preserve the sequence of the
original experiments; they are not a package API.

| Stage | Script | Purpose |
|---|---|---|
| 01 | `01_audit.py` | Schema, header and class-count audit |
| 02 | `02_prepare_sample.py` | Deterministic benign-train and mixed-evaluation samples |
| 03 | `03_preprocess.py` | Training-only imputation/scaling and feature summary |
| 04 | `04_run_models.py` | k-means and custom Isolation Forest |
| 05 | `05_stability.py` | Fixed-seed stability checks |
| 06-07 | `06_run_novelty_baselines.py`, `07_novelty_stability.py` | One-Class SVM, PCA, ECOD and baseline stability |
| 08 | `08_run_hdbscan_analysis.py` | Exploratory clustering of mixed data |
| 09-11 | `09_prepare_supervised_sample.py`, `10_run_random_forest.py`, `11_run_hist_gradient_boosting.py` | Supervised train/validation split and benchmarks |
| 12-13 | `12_run_clustering_extensions.py`, `13_run_clustering_refined_sweep.py` | BIRCH, GMM, DBSCAN, OPTICS and local parameter sweeps |
| 14 | `14_unknown_attack_evaluation.py` | Leave-one-attack-class-out comparison |
| 15-16 | `15_supervised_benign_scale_study.py`, `16_clustering_train_scale_study.py` | Training-size studies |
| 17-18 | `17_benign_augmentation_study.py`, `18_birch_benign_augmentation_sweep.py` | Benign augmentation experiments |
| 19-21 | `19_source_aware_clustering.py`, `20_benign_source_balancing_study.py`, `21_birch_source_balanced_sweep.py` | Source-aware scoring, balancing and BIRCH sweeps |
| 22 | `22_birch_source_balanced_ctgan.py` | Optional CTGAN augmentation study |

Stages 09-11 need additional sampled training/validation data. Later scale and
source-aware stages can require substantial memory and processing time. Check
the script before starting a full sweep.

For stage 22 only:

```bash
python -m pip install -r requirements-ctgan.txt
python python/22_birch_source_balanced_ctgan.py
```

CTGAN is optional and is not exercised by the default CI job. The requirements
file pins the directly used versions, not every transitive dependency.

Shared code:

- [`data_utils.py`](data_utils.py): data audit, sampling, preprocessing and metrics;
- [`clustering_stage.py`](clustering_stage.py): clustering evaluation;
- [`novelty_stage.py`](novelty_stage.py): novelty-detection baselines;
- [`supervised_stage.py`](supervised_stage.py): supervised evaluation;
- [`simple_iforest.py`](simple_iforest.py): custom Isolation Forest implementation.
