from __future__ import annotations

import importlib.util
import pandas as pd
from pathlib import Path

from data_utils import RESULTS_DIR, load_samples, prepare_features


STABILITY_SEEDS = [2027, 2028, 2029]

RUN_MODELS_PATH = Path(__file__).with_name("04_run_models.py")
RUN_MODELS_SPEC = importlib.util.spec_from_file_location("run_models_module", RUN_MODELS_PATH)
if RUN_MODELS_SPEC is None or RUN_MODELS_SPEC.loader is None:
    raise RuntimeError("Nie udalo sie zaladowac 04_run_models.py.")
run_models = importlib.util.module_from_spec(RUN_MODELS_SPEC)
RUN_MODELS_SPEC.loader.exec_module(run_models)

KMEANS_CONFIGS = run_models.KMEANS_CONFIGS
IFOREST_CONFIGS = run_models.IFOREST_CONFIGS
evaluate_kmeans = run_models.evaluate_kmeans
evaluate_iforest = run_models.evaluate_iforest


def main() -> None:
    """Check how the final configurations change across a few fixed seeds."""
    train, eval_frame = load_samples()
    x_train, x_eval, _ = prepare_features(train, eval_frame)

    rows: list[dict[str, object]] = []

    for seed in STABILITY_SEEDS:
        kmeans_cache = {}
        for config in KMEANS_CONFIGS:
            metrics, _ = evaluate_kmeans(
                x_train,
                x_eval,
                eval_frame,
                config,
                kmeans_cache,
                random_seed=seed,
            )
            rows.append(
                {
                    "algorithm": "kmeans",
                    "config_name": config["name"],
                    "seed": seed,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                }
            )

        for config in IFOREST_CONFIGS:
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
    details_path = RESULTS_DIR / "python_stability_details.csv"
    summary_path = RESULTS_DIR / "python_stability_summary.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Zapisano:")
    print(details_path)
    print(summary_path)


if __name__ == "__main__":
    main()
