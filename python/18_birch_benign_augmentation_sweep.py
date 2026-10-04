from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from clustering_stage import per_class_recall, run_algorithm_config
from data_utils import (
    MODEL_RANDOM_SEED,
    RESULTS_DIR,
    load_clustering_samples_with_train_size,
    prepare_features,
)


BASE_BENIGN_ROWS = 100_000
SYNTHETIC_ROWS = 50_000
GMM_COMPONENTS = 32
GMM_COVARIANCE_TYPE = "tied"
GMM_REG_COVAR = 1e-6
SCORE_PERCENTILES = [93.0, 94.0, 95.0, 96.0, 97.0]


def main() -> None:
    """Refine the most promising benign-augmentation setting for Birch."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_frame, eval_frame = load_clustering_samples_with_train_size(BASE_BENIGN_ROWS)
    x_train, x_eval, _ = prepare_features(train_frame, eval_frame)

    generator = GaussianMixture(
        n_components=GMM_COMPONENTS,
        covariance_type=GMM_COVARIANCE_TYPE,
        reg_covar=GMM_REG_COVAR,
        random_state=MODEL_RANDOM_SEED,
        max_iter=300,
    )
    generator.fit(x_train)
    synthetic, _ = generator.sample(SYNTHETIC_ROWS)
    synthetic = np.clip(synthetic, x_train.min(axis=0), x_train.max(axis=0))
    x_train_aug = np.vstack([x_train, synthetic])

    rows: list[dict[str, str | float | int]] = []
    recall_rows: list[dict[str, str | float | int]] = []

    for percentile in SCORE_PERCENTILES:
        config = {
            "name": f"birch_gmm50k_p{int(percentile)}",
            "threshold": 0.75,
            "branching_factor": 50,
            "score_percentile": percentile,
        }
        metrics, predictions = run_algorithm_config("birch", x_train_aug, x_eval, eval_frame, config)
        metrics["augmentation_name"] = "gmm_sample_50k"
        metrics["synthetic_rows"] = SYNTHETIC_ROWS
        metrics["train_rows_total"] = int(x_train_aug.shape[0])
        rows.append(metrics)

        for row in per_class_recall(eval_frame, predictions, str(config["name"])):
            row["augmentation_name"] = "gmm_sample_50k"
            row["synthetic_rows"] = SYNTHETIC_ROWS
            row["train_rows_total"] = int(x_train_aug.shape[0])
            recall_rows.append(row)

    summary = pd.DataFrame(rows)
    recall_by_class = pd.DataFrame(recall_rows)

    summary_path = RESULTS_DIR / "python_birch_benign_augmentation_sweep.csv"
    recall_path = RESULTS_DIR / "python_birch_benign_augmentation_sweep_recall_by_class.csv"
    summary.to_csv(summary_path, index=False)
    recall_by_class.to_csv(recall_path, index=False)

    print("Zapisano sweep BIRCH dla augmentacji Benign przez GMM:")
    print(summary_path)
    print(recall_path)
    print()
    print(
        summary[["config_name", "precision", "recall", "f1", "fpr", "runtime_seconds"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
