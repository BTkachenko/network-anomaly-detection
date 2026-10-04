from __future__ import annotations

import pandas as pd

from data_utils import MODEL_RANDOM_SEED, RESULTS_DIR, load_samples, prepare_features
from novelty_stage import (
    ECOD_CONFIGS,
    IFOREST_NOVELTY_CONFIGS,
    OCSVM_CONFIGS,
    evaluate_iforest,
    evaluate_ocsvm,
    PCA_CONFIGS,
    evaluate_ecod,
    evaluate_pca_reconstruction,
)


STABILITY_SEEDS = [MODEL_RANDOM_SEED, MODEL_RANDOM_SEED + 1, MODEL_RANDOM_SEED + 2]


def main() -> None:
    """Check how the novelty-baseline results change across a few fixed seeds."""
    train, eval_frame = load_samples()
    x_train, x_eval, _ = prepare_features(train, eval_frame)

    rows: list[dict[str, object]] = []

    for seed in STABILITY_SEEDS:
        for config in IFOREST_NOVELTY_CONFIGS:
            metrics, _ = evaluate_iforest(
                x_train,
                x_eval,
                eval_frame,
                config,
                random_seed=seed,
            )
            rows.append(
                {
                    "algorithm": "isolation_forest",
                    "config_name": config["name"],
                    "seed": seed,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                }
            )

        for config in OCSVM_CONFIGS:
            metrics, _ = evaluate_ocsvm(
                x_train,
                x_eval,
                eval_frame,
                config,
                random_seed=seed,
            )
            rows.append(
                {
                    "algorithm": "one_class_svm",
                    "config_name": config["name"],
                    "seed": seed,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                }
            )

        for config in PCA_CONFIGS:
            metrics, _ = evaluate_pca_reconstruction(
                x_train,
                x_eval,
                eval_frame,
                config,
                random_seed=seed,
            )
            rows.append(
                {
                    "algorithm": "pca_reconstruction",
                    "config_name": config["name"],
                    "seed": seed,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                }
            )

        for config in ECOD_CONFIGS:
            metrics, _ = evaluate_ecod(
                x_train,
                x_eval,
                eval_frame,
                config,
                random_seed=seed,
            )
            rows.append(
                {
                    "algorithm": "ecod",
                    "config_name": config["name"],
                    "seed": seed,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                }
            )

    details = pd.DataFrame(rows)
    summary = (
        details.groupby(["algorithm", "config_name"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
        )
        .sort_values(["algorithm", "config_name"])
        .reset_index(drop=True)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    details_path = RESULTS_DIR / "python_novelty_stability_details.csv"
    summary_path = RESULTS_DIR / "python_novelty_stability_summary.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Zapisano:")
    print(details_path)
    print(summary_path)


if __name__ == "__main__":
    main()
