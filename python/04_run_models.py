from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from data_utils import MODEL_RANDOM_SEED, RESULTS_DIR, load_samples, metrics_table, prepare_features
from simple_iforest import SimpleIsolationForest


KMEANS_CONFIGS = [
    {"name": "kmeans_practical", "k": 256, "threshold_percentile": 91.0},
    {"name": "kmeans_mid_fpr", "k": 1024, "threshold_percentile": 90.0},
    {"name": "kmeans_max_f1", "k": 1024, "threshold_percentile": 84.0},
]

IFOREST_CONFIGS = [
    {"name": "iforest_practical", "trees": 100, "sample_size": 256, "threshold_percentile": 95.0},
    {"name": "iforest_balanced", "trees": 200, "sample_size": 512, "threshold_percentile": 92.0},
    {"name": "iforest_higher_recall", "trees": 300, "sample_size": 512, "threshold_percentile": 88.0},
]

KMEANS_COMPLEXITY = [
    {"algorithm": "kmeans", "setting": "k=256", "k": 256, "threshold_percentile": 90.0},
    {"algorithm": "kmeans", "setting": "k=512", "k": 512, "threshold_percentile": 90.0},
    {"algorithm": "kmeans", "setting": "k=1024", "k": 1024, "threshold_percentile": 90.0},
]

IFOREST_COMPLEXITY = [
    {
        "algorithm": "isolation_forest",
        "setting": "trees=100,sample=256",
        "trees": 100,
        "sample_size": 256,
        "threshold_percentile": 90.0,
    },
    {
        "algorithm": "isolation_forest",
        "setting": "trees=200,sample=512",
        "trees": 200,
        "sample_size": 512,
        "threshold_percentile": 90.0,
    },
    {
        "algorithm": "isolation_forest",
        "setting": "trees=300,sample=512",
        "trees": 300,
        "sample_size": 512,
        "threshold_percentile": 90.0,
    },
]

KMEANS_THRESHOLDS = [
    {"algorithm": "kmeans", "setting": "k=1024", "k": 1024, "threshold_percentile": value}
    for value in (95.0, 92.0, 90.0, 88.0, 86.0, 84.0)
]

IFOREST_THRESHOLDS = [
    {
        "algorithm": "isolation_forest",
        "setting": "trees=300,sample=512",
        "trees": 300,
        "sample_size": 512,
        "threshold_percentile": value,
    }
    for value in (95.0, 92.0, 90.0, 88.0, 86.0, 84.0)
]


def per_class_recall(
    eval_frame: pd.DataFrame, predictions: np.ndarray, config_name: str
) -> list[dict[str, str | float]]:
    """Compute recall for each attack class separately."""
    rows: list[dict[str, str | float]] = []
    attack_only = eval_frame.loc[eval_frame["binary_label"].astype(int) == 1].copy()
    attack_only["prediction"] = predictions[eval_frame["binary_label"].astype(int).to_numpy() == 1]

    for label, frame in attack_only.groupby("coarse_label", sort=True):
        truth = frame["binary_label"].astype(int).to_numpy()
        pred = frame["prediction"].astype(int).to_numpy()
        tp = int(((truth == 1) & (pred == 1)).sum())
        fn = int(((truth == 1) & (pred == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({"config_name": config_name, "attack_class": label, "recall": recall})
    return rows


def evaluate_kmeans(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict,
    cache: dict[int, tuple[KMeans, np.ndarray, float]],
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[dict, np.ndarray]:
    """Evaluate one k-means configuration."""
    cache_key = (config["k"], random_seed)
    if cache_key not in cache:
        started = time.perf_counter()
        model = KMeans(n_clusters=config["k"], random_state=random_seed, n_init=10)
        model.fit(x_train)
        train_distances = np.linalg.norm(x_train - model.cluster_centers_[model.labels_], axis=1)
        cache[cache_key] = (model, train_distances, time.perf_counter() - started)

    model, train_distances, fit_seconds = cache[cache_key]
    threshold = float(np.percentile(train_distances, config["threshold_percentile"]))
    labels = model.predict(x_eval)
    eval_distances = np.linalg.norm(x_eval - model.cluster_centers_[labels], axis=1)
    predictions = (eval_distances > threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)

    row = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fpr": metrics["fpr"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "runtime_seconds": fit_seconds,
    }
    return row, predictions


def evaluate_iforest(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict,
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[dict, np.ndarray]:
    """Evaluate one Isolation Forest configuration."""
    started = time.perf_counter()
    model = SimpleIsolationForest(
        n_trees=config["trees"],
        sample_size=config["sample_size"],
        random_state=random_seed,
    )
    model.fit(x_train)
    train_scores = model.score_samples(x_train)
    threshold = float(np.percentile(train_scores, config["threshold_percentile"]))
    predictions = (model.score_samples(x_eval) > threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)

    row = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fpr": metrics["fpr"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def run_final_configs(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the three final configurations per algorithm."""
    experiment_rows: list[dict] = []
    recall_rows: list[dict] = []
    kmeans_cache: dict[tuple[int, int], tuple[KMeans, np.ndarray, float]] = {}

    for config in KMEANS_CONFIGS:
        metrics, predictions = evaluate_kmeans(x_train, x_eval, eval_frame, config, kmeans_cache)
        experiment_rows.append(
            {
                "algorithm": "kmeans",
                "config_name": config["name"],
                "parameters": f"k={config['k']}, threshold_percentile={config['threshold_percentile']}",
                **metrics,
            }
        )
        recall_rows.extend(per_class_recall(eval_frame, predictions, config["name"]))

    for config in IFOREST_CONFIGS:
        metrics, predictions = evaluate_iforest(x_train, x_eval, eval_frame, config)
        experiment_rows.append(
            {
                "algorithm": "isolation_forest",
                "config_name": config["name"],
                "parameters": (
                    f"trees={config['trees']}, sample_size={config['sample_size']}, "
                    f"threshold_percentile={config['threshold_percentile']}"
                ),
                **metrics,
            }
        )
        recall_rows.extend(per_class_recall(eval_frame, predictions, config["name"]))

    return pd.DataFrame(experiment_rows), pd.DataFrame(recall_rows)


def run_trends(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the trend tables used in the report."""
    parameter_rows: list[dict] = []
    complexity_rows: list[dict] = []
    kmeans_cache: dict[tuple[int, int], tuple[KMeans, np.ndarray, float]] = {}

    for config in KMEANS_THRESHOLDS:
        metrics, _ = evaluate_kmeans(x_train, x_eval, eval_frame, config, kmeans_cache)
        parameter_rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "threshold_percentile": config["threshold_percentile"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )

    for config in IFOREST_THRESHOLDS:
        metrics, _ = evaluate_iforest(x_train, x_eval, eval_frame, config)
        parameter_rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "threshold_percentile": config["threshold_percentile"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )

    for config in KMEANS_COMPLEXITY:
        metrics, _ = evaluate_kmeans(x_train, x_eval, eval_frame, config, kmeans_cache)
        complexity_rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "threshold_percentile": config["threshold_percentile"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )

    for config in IFOREST_COMPLEXITY:
        metrics, _ = evaluate_iforest(x_train, x_eval, eval_frame, config)
        complexity_rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "threshold_percentile": config["threshold_percentile"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )

    return pd.DataFrame(parameter_rows), pd.DataFrame(complexity_rows)


def main() -> None:
    """Run both algorithms on the shared sample and save the report tables."""
    parser = argparse.ArgumentParser(description="Run k-means and Isolation Forest on the shared sample.")
    parser.add_argument(
        "--only",
        choices=["all", "kmeans", "iforest"],
        default="all",
        help="Run all models or only one algorithm family.",
    )
    args = parser.parse_args()

    train, eval_frame = load_samples()
    x_train, x_eval, used_columns = prepare_features(train, eval_frame)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.only == "all":
        experiments, recall_by_class = run_final_configs(x_train, x_eval, eval_frame)
        parameter_trend, complexity_trend = run_trends(x_train, x_eval, eval_frame)

        experiments.insert(0, "feature_count", len(used_columns))
        experiments.to_csv(RESULTS_DIR / "python_experiments.csv", index=False)
        recall_by_class.to_csv(RESULTS_DIR / "python_recall_by_class.csv", index=False)
        parameter_trend.to_csv(RESULTS_DIR / "python_parameter_trend.csv", index=False)
        complexity_trend.to_csv(RESULTS_DIR / "python_complexity_trend.csv", index=False)

        print(experiments.sort_values(["algorithm", "f1"], ascending=[True, False]).to_string(index=False))
        return

    rows = []
    kmeans_cache: dict[int, tuple[KMeans, np.ndarray, float]] = {}

    if args.only == "kmeans":
        for config in KMEANS_CONFIGS:
            metrics, _ = evaluate_kmeans(x_train, x_eval, eval_frame, config, kmeans_cache)
            rows.append({"algorithm": "kmeans", "config_name": config["name"], **metrics})
    elif args.only == "iforest":
        for config in IFOREST_CONFIGS:
            metrics, _ = evaluate_iforest(x_train, x_eval, eval_frame, config)
            rows.append({"algorithm": "isolation_forest", "config_name": config["name"], **metrics})

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
