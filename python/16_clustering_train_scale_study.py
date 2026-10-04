from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from clustering_stage import per_class_recall, run_algorithm_config
from data_utils import (
    MODEL_RANDOM_SEED,
    RESULTS_DIR,
    load_clustering_samples_with_train_size,
    metrics_table,
    prepare_features,
    sample_summary,
)


DEFAULT_BENIGN_SIZES = [30_000, 50_000, 100_000]

KMEANS_CONFIGS = [
    {"name": "kmeans_mid_fpr", "k": 1024, "threshold_percentile": 90.0},
    {"name": "kmeans_max_f1", "k": 1024, "threshold_percentile": 84.0},
]

BIRCH_CONFIGS = [
    {"name": "birch_refined_t0.75_p94", "threshold": 0.75, "branching_factor": 50, "score_percentile": 94.0},
    {"name": "birch_refined_t0.75_p91", "threshold": 0.75, "branching_factor": 50, "score_percentile": 91.0},
]

DBSCAN_CONFIGS = [
    {"name": "dbscan_refined_eps0.38_m8", "eps": 0.38, "eps_percentile": "manual_local", "min_samples": 8},
    {"name": "dbscan_refined_eps0.26_m12", "eps": 0.26, "eps_percentile": "manual_local", "min_samples": 12},
]

GMM_CONFIGS = [
    {
        "name": "gmm_refined_k32_tied_p10",
        "n_components": 32,
        "covariance_type": "tied",
        "reg_covar": 1e-6,
        "lower_percentile": 10.0,
    }
]

OPTICS_CONFIGS = [
    {"name": "optics_balanced", "max_eps": 0.500927238311737, "eps_percentile": 85, "min_samples": 10},
]


def parse_sizes(values: list[str]) -> list[int]:
    """Parse requested benign train sizes."""
    sizes = sorted({int(value) for value in values})
    if not sizes:
        raise ValueError("Lista rozmiarow nie moze byc pusta.")
    return sizes


def evaluate_kmeans_config(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Evaluate one fixed k-means configuration."""
    started = time.perf_counter()
    model = KMeans(n_clusters=int(config["k"]), random_state=random_seed, n_init=10)
    model.fit(x_train)
    train_distances = np.linalg.norm(x_train - model.cluster_centers_[model.labels_], axis=1)
    threshold = float(np.percentile(train_distances, float(config["threshold_percentile"])))
    labels = model.predict(x_eval)
    eval_distances = np.linalg.norm(x_eval - model.cluster_centers_[labels], axis=1)
    predictions = (eval_distances > threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)

    row: dict[str, float | int | str] = {
        "algorithm": "kmeans",
        "config_name": str(config["name"]),
        "parameters": f"k={config['k']}, threshold_percentile={config['threshold_percentile']}",
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def build_scale_summary(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize best points per algorithm and train size."""
    rows: list[dict[str, str | int | float]] = []

    for (algorithm, train_benign_rows), group in raw_results.groupby(
        ["algorithm", "train_benign_rows"], sort=True
    ):
        best_f1 = group.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
        rows.append(
            {
                "algorithm": algorithm,
                "train_benign_rows": int(train_benign_rows),
                "selection_rule": "best_f1",
                "config_name": str(best_f1["config_name"]),
                "precision": float(best_f1["precision"]),
                "recall": float(best_f1["recall"]),
                "f1": float(best_f1["f1"]),
                "fpr": float(best_f1["fpr"]),
                "runtime_seconds": float(best_f1["runtime_seconds"]),
            }
        )

        for limit in (0.12, 0.15):
            practical = group.loc[group["fpr"] <= limit].copy()
            if practical.empty:
                continue
            best_practical = practical.sort_values(
                ["recall", "f1", "precision"], ascending=[False, False, False]
            ).iloc[0]
            rows.append(
                {
                    "algorithm": algorithm,
                    "train_benign_rows": int(train_benign_rows),
                    "selection_rule": f"best_recall_fpr_le_{limit:.2f}",
                    "config_name": str(best_practical["config_name"]),
                    "precision": float(best_practical["precision"]),
                    "recall": float(best_practical["recall"]),
                    "f1": float(best_practical["f1"]),
                    "fpr": float(best_practical["fpr"]),
                    "runtime_seconds": float(best_practical["runtime_seconds"]),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Compare clustering results for larger benign training samples."""
    parser = argparse.ArgumentParser(
        description="Compare clustering models for benign train sizes 30k / 50k / 100k."
    )
    parser.add_argument(
        "--benign-sizes",
        nargs="+",
        default=[str(size) for size in DEFAULT_BENIGN_SIZES],
        help="Lista rozmiarow benign train do porownania, np. 30000 50000 100000.",
    )
    args = parser.parse_args()

    benign_sizes = parse_sizes(args.benign_sizes)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, str | int | float]] = []
    recall_rows: list[dict[str, str | int | float]] = []
    sample_rows: list[pd.DataFrame] = []

    for benign_rows in benign_sizes:
        train_frame, eval_frame = load_clustering_samples_with_train_size(benign_rows)
        x_train, x_eval, used_columns = prepare_features(train_frame, eval_frame)

        sample_info = sample_summary(train_frame, eval_frame).copy()
        sample_info.insert(0, "train_benign_rows", benign_rows)
        sample_rows.append(sample_info)

        for config in KMEANS_CONFIGS:
            metrics, predictions = evaluate_kmeans_config(x_train, x_eval, eval_frame, config)
            metrics["train_benign_rows"] = benign_rows
            metrics["feature_count"] = len(used_columns)
            raw_rows.append(metrics)
            for row in per_class_recall(eval_frame, predictions, str(config["name"])):
                row["train_benign_rows"] = benign_rows
                row["algorithm"] = "kmeans"
                recall_rows.append(row)

        for algorithm, configs in (
            ("birch", BIRCH_CONFIGS),
            ("gmm", GMM_CONFIGS),
            ("dbscan", DBSCAN_CONFIGS),
            ("optics", OPTICS_CONFIGS),
        ):
            for config in configs:
                metrics, predictions = run_algorithm_config(algorithm, x_train, x_eval, eval_frame, config)
                metrics["train_benign_rows"] = benign_rows
                metrics["feature_count"] = len(used_columns)
                raw_rows.append(metrics)
                for row in per_class_recall(eval_frame, predictions, str(config["name"])):
                    row["train_benign_rows"] = benign_rows
                    row["algorithm"] = algorithm
                    recall_rows.append(row)

    raw_results = pd.DataFrame(raw_rows)
    recall_by_class = pd.DataFrame(recall_rows)
    sample_results = pd.concat(sample_rows, ignore_index=True)
    summary = build_scale_summary(raw_results)

    raw_path = RESULTS_DIR / "python_clustering_train_scale_study.csv"
    summary_path = RESULTS_DIR / "python_clustering_train_scale_summary.csv"
    recall_path = RESULTS_DIR / "python_clustering_train_scale_recall_by_class.csv"
    sample_path = RESULTS_DIR / "python_clustering_train_scale_sample_summary.csv"

    raw_results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    recall_by_class.to_csv(recall_path, index=False)
    sample_results.to_csv(sample_path, index=False)

    print("Zapisano porownanie klasteryzacji dla roznych rozmiarow benign train:")
    print(raw_path)
    print(summary_path)
    print(recall_path)
    print(sample_path)
    print()
    print(summary.sort_values(["algorithm", "train_benign_rows", "selection_rule"]).to_string(index=False))


if __name__ == "__main__":
    main()
