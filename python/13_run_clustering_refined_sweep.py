from __future__ import annotations

import pandas as pd

from clustering_stage import (
    estimate_density_reference,
    per_class_recall,
    run_algorithm_config,
    summarize_grid,
)
from data_utils import RESULTS_DIR, load_samples, prepare_features


def build_birch_refined_configs() -> list[dict[str, float | int | str]]:
    """Local sweep around the most promising Birch region."""
    configs: list[dict[str, float | int | str]] = []
    for threshold in (0.75, 0.85, 0.95, 1.05, 1.15):
        for percentile in (91.0, 92.0, 93.0, 94.0):
            configs.append(
                {
                    "name": f"birch_refined_t{threshold:.2f}_p{int(percentile)}",
                    "threshold": threshold,
                    "branching_factor": 50,
                    "score_percentile": percentile,
                }
            )
    return configs


def build_dbscan_refined_configs() -> list[dict[str, float | int | str]]:
    """Local sweep around the eps band that looked best in the first round."""
    configs: list[dict[str, float | int | str]] = []
    for eps in (0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50):
        for min_samples in (8, 10, 12):
            configs.append(
                {
                    "name": f"dbscan_refined_eps{eps:.2f}_m{min_samples}",
                    "eps": eps,
                    "eps_percentile": "manual_local",
                    "min_samples": min_samples,
                }
            )
    return configs


def build_gmm_refined_configs() -> list[dict[str, float | int | str]]:
    """One last targeted sweep for GMM around the strongest previous region."""
    configs: list[dict[str, float | int | str]] = []
    for n_components in (16, 24, 32):
        for covariance_type in ("diag", "tied"):
            for lower_percentile in (5.0, 8.0, 10.0):
                configs.append(
                    {
                        "name": f"gmm_refined_k{n_components}_{covariance_type}_p{int(lower_percentile)}",
                        "n_components": n_components,
                        "covariance_type": covariance_type,
                        "reg_covar": 1e-6,
                        "lower_percentile": lower_percentile,
                    }
                )
    return configs


def evaluate_grid(
    algorithm: str,
    configs: list[dict[str, float | int | str]],
    x_train,
    x_eval,
    eval_frame: pd.DataFrame,
    feature_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a full grid for one algorithm and collect per-class recalls."""
    rows: list[dict[str, str | float | int]] = []
    recall_rows: list[dict[str, str | float]] = []

    for config in configs:
        metrics, predictions = run_algorithm_config(algorithm, x_train, x_eval, eval_frame, config)
        metrics["feature_count"] = feature_count
        rows.append(metrics)
        recall_rows.extend(per_class_recall(eval_frame, predictions, str(config["name"])))

    return pd.DataFrame(rows), pd.DataFrame(recall_rows)


def main() -> None:
    """Run short local sweeps for the clustering methods that still need closure."""
    train, eval_frame = load_samples()
    x_train, x_eval, used_columns = prepare_features(train, eval_frame)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    density_reference = estimate_density_reference(x_train, neighbor_count=10)
    density_reference.to_csv(RESULTS_DIR / "python_clustering_density_reference.csv", index=False)

    birch_grid, birch_recall = evaluate_grid(
        "birch",
        build_birch_refined_configs(),
        x_train,
        x_eval,
        eval_frame,
        len(used_columns),
    )
    dbscan_grid, dbscan_recall = evaluate_grid(
        "dbscan",
        build_dbscan_refined_configs(),
        x_train,
        x_eval,
        eval_frame,
        len(used_columns),
    )
    gmm_grid, gmm_recall = evaluate_grid(
        "gmm",
        build_gmm_refined_configs(),
        x_train,
        x_eval,
        eval_frame,
        len(used_columns),
    )

    birch_grid.to_csv(RESULTS_DIR / "python_birch_refined_grid.csv", index=False)
    dbscan_grid.to_csv(RESULTS_DIR / "python_dbscan_refined_grid.csv", index=False)
    gmm_grid.to_csv(RESULTS_DIR / "python_gmm_refined_grid.csv", index=False)

    pd.concat([birch_recall, dbscan_recall, gmm_recall], ignore_index=True).to_csv(
        RESULTS_DIR / "python_clustering_refined_recall_by_class.csv",
        index=False,
    )

    summary = pd.concat(
        [
            summarize_grid(birch_grid, "birch"),
            summarize_grid(dbscan_grid, "dbscan"),
            summarize_grid(gmm_grid, "gmm"),
        ],
        ignore_index=True,
    )
    summary.to_csv(RESULTS_DIR / "python_clustering_refined_summary.csv", index=False)

    print("Refined Birch:")
    print(birch_grid.sort_values(["fpr", "recall"], ascending=[True, False]).head(8).to_string(index=False))
    print("\nRefined DBSCAN:")
    print(dbscan_grid.sort_values(["fpr", "recall"], ascending=[True, False]).head(8).to_string(index=False))
    print("\nRefined GMM:")
    print(gmm_grid.sort_values(["fpr", "recall"], ascending=[True, False]).head(8).to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
