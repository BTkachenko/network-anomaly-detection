from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from clustering_stage import per_class_recall, run_algorithm_config
from data_utils import (
    MODEL_RANDOM_SEED,
    RESULTS_DIR,
    load_clustering_samples_with_train_size,
    metrics_table,
    prepare_features,
)


BASE_BENIGN_ROWS = 100_000

AUGMENTATION_CONFIGS = [
    {
        "name": "baseline_100k",
        "method": "none",
        "synthetic_rows": 0,
        "neighbor_count": 6,
        "beta_alpha": 0.5,
        "noise_std": 0.00,
    },
    {
        "name": "local_interp_25k",
        "method": "local_interp",
        "synthetic_rows": 25_000,
        "neighbor_count": 6,
        "beta_alpha": 0.5,
        "noise_std": 0.01,
    },
    {
        "name": "local_interp_50k",
        "method": "local_interp",
        "synthetic_rows": 50_000,
        "neighbor_count": 6,
        "beta_alpha": 0.5,
        "noise_std": 0.01,
    },
    {
        "name": "gmm_sample_25k",
        "method": "gmm_sample",
        "synthetic_rows": 25_000,
        "gmm_components": 32,
        "gmm_covariance_type": "tied",
        "gmm_reg_covar": 1e-6,
    },
    {
        "name": "gmm_sample_50k",
        "method": "gmm_sample",
        "synthetic_rows": 50_000,
        "gmm_components": 32,
        "gmm_covariance_type": "tied",
        "gmm_reg_covar": 1e-6,
    },
]

KMEANS_CONFIG = {"name": "kmeans_mid_fpr", "k": 1024, "threshold_percentile": 90.0}
BIRCH_CONFIG = {
    "name": "birch_refined_t0.75_p94",
    "threshold": 0.75,
    "branching_factor": 50,
    "score_percentile": 94.0,
}
DBSCAN_CONFIG = {
    "name": "dbscan_refined_eps0.38_m8",
    "eps": 0.38,
    "eps_percentile": "manual_local",
    "min_samples": 8,
}


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


def generate_local_interpolation_benign(
    x_train: np.ndarray,
    synthetic_rows: int,
    neighbor_count: int,
    beta_alpha: float,
    noise_std: float,
    seed: int = MODEL_RANDOM_SEED,
) -> np.ndarray:
    """Generate synthetic benign rows by interpolating local benign neighbours."""
    if synthetic_rows <= 0:
        return np.empty(shape=(0, x_train.shape[1]), dtype=float)

    rng = np.random.default_rng(seed)
    knn = NearestNeighbors(n_neighbors=neighbor_count, n_jobs=-1)
    knn.fit(x_train)
    neighbor_idx = knn.kneighbors(return_distance=False)

    base_idx = rng.integers(low=0, high=x_train.shape[0], size=synthetic_rows)
    neighbor_slot = rng.integers(low=1, high=neighbor_count, size=synthetic_rows)
    partner_idx = neighbor_idx[base_idx, neighbor_slot]

    base = x_train[base_idx]
    partner = x_train[partner_idx]
    lam = rng.beta(beta_alpha, beta_alpha, size=(synthetic_rows, 1))
    synthetic = base + lam * (partner - base)

    if noise_std > 0:
        synthetic += rng.normal(loc=0.0, scale=noise_std, size=synthetic.shape)

    feature_min = x_train.min(axis=0)
    feature_max = x_train.max(axis=0)
    synthetic = np.clip(synthetic, feature_min, feature_max)
    return synthetic


def generate_gmm_benign(
    x_train: np.ndarray,
    synthetic_rows: int,
    n_components: int,
    covariance_type: str,
    reg_covar: float,
    seed: int = MODEL_RANDOM_SEED,
) -> np.ndarray:
    """Generate synthetic benign rows by sampling from a GMM fitted on benign train."""
    if synthetic_rows <= 0:
        return np.empty(shape=(0, x_train.shape[1]), dtype=float)

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=seed,
        max_iter=300,
    )
    model.fit(x_train)
    synthetic, _ = model.sample(synthetic_rows)
    feature_min = x_train.min(axis=0)
    feature_max = x_train.max(axis=0)
    return np.clip(synthetic, feature_min, feature_max)


def main() -> None:
    """Run the first benign augmentation study for the clustering pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_frame, eval_frame = load_clustering_samples_with_train_size(BASE_BENIGN_ROWS)
    x_train, x_eval, used_columns = prepare_features(train_frame, eval_frame)

    raw_rows: list[dict[str, str | int | float]] = []
    recall_rows: list[dict[str, str | int | float]] = []

    for augment_config in AUGMENTATION_CONFIGS:
        method = str(augment_config["method"])
        if method == "none":
            synthetic = np.empty(shape=(0, x_train.shape[1]), dtype=float)
        elif method == "local_interp":
            synthetic = generate_local_interpolation_benign(
                x_train=x_train,
                synthetic_rows=int(augment_config["synthetic_rows"]),
                neighbor_count=int(augment_config["neighbor_count"]),
                beta_alpha=float(augment_config["beta_alpha"]),
                noise_std=float(augment_config["noise_std"]),
            )
        elif method == "gmm_sample":
            synthetic = generate_gmm_benign(
                x_train=x_train,
                synthetic_rows=int(augment_config["synthetic_rows"]),
                n_components=int(augment_config["gmm_components"]),
                covariance_type=str(augment_config["gmm_covariance_type"]),
                reg_covar=float(augment_config["gmm_reg_covar"]),
            )
        else:
            raise ValueError(f"Nieznana metoda augmentacji: {method}")

        x_train_aug = np.vstack([x_train, synthetic]) if len(synthetic) else x_train.copy()

        for algorithm, config in (
            ("kmeans", KMEANS_CONFIG),
            ("birch", BIRCH_CONFIG),
            ("dbscan", DBSCAN_CONFIG),
        ):
            if algorithm == "kmeans":
                metrics, predictions = evaluate_kmeans_config(x_train_aug, x_eval, eval_frame, config)
            else:
                metrics, predictions = run_algorithm_config(
                    algorithm, x_train_aug, x_eval, eval_frame, config
                )

            metrics["augmentation_name"] = str(augment_config["name"])
            metrics["augmentation_method"] = method
            metrics["synthetic_rows"] = int(augment_config["synthetic_rows"])
            metrics["train_rows_total"] = int(x_train_aug.shape[0])
            metrics["neighbor_count"] = int(augment_config.get("neighbor_count", 0))
            metrics["beta_alpha"] = float(augment_config.get("beta_alpha", 0.0))
            metrics["noise_std"] = float(augment_config.get("noise_std", 0.0))
            metrics["gmm_components"] = int(augment_config.get("gmm_components", 0))
            metrics["gmm_reg_covar"] = float(augment_config.get("gmm_reg_covar", 0.0))
            metrics["feature_count"] = len(used_columns)
            raw_rows.append(metrics)

            for row in per_class_recall(eval_frame, predictions, str(config["name"])):
                row["algorithm"] = algorithm
                row["augmentation_name"] = str(augment_config["name"])
                row["augmentation_method"] = method
                row["synthetic_rows"] = int(augment_config["synthetic_rows"])
                row["train_rows_total"] = int(x_train_aug.shape[0])
                recall_rows.append(row)

    raw_results = pd.DataFrame(raw_rows)
    recall_by_class = pd.DataFrame(recall_rows)

    summary = raw_results[
        [
            "algorithm",
            "config_name",
            "augmentation_name",
            "augmentation_method",
            "synthetic_rows",
            "train_rows_total",
            "precision",
            "recall",
            "f1",
            "fpr",
            "runtime_seconds",
        ]
    ].sort_values(["algorithm", "synthetic_rows"])

    raw_path = RESULTS_DIR / "python_benign_augmentation_study.csv"
    recall_path = RESULTS_DIR / "python_benign_augmentation_recall_by_class.csv"
    summary_path = RESULTS_DIR / "python_benign_augmentation_summary.csv"

    raw_results.to_csv(raw_path, index=False)
    recall_by_class.to_csv(recall_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Zapisano wyniki augmentacji klasy Benign dla klasteryzacji:")
    print(raw_path)
    print(recall_path)
    print(summary_path)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
