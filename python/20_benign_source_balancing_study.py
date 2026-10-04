from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import Birch, KMeans
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from clustering_stage import distance_to_centers, evaluate_dbscan, per_class_recall
from data_utils import (
    MODEL_RANDOM_SEED,
    RESULTS_DIR,
    load_clustering_samples_with_train_size,
    metrics_table,
    prepare_features,
)


TRAIN_BENIGN_ROWS = 100_000


@dataclass(frozen=True)
class AugmentConfig:
    name: str
    method: str


AUGMENT_CONFIGS = [
    AugmentConfig("baseline_100k", "none"),
    AugmentConfig("source_balanced_local_interp", "local_interp"),
    AugmentConfig("source_balanced_gmm", "gmm"),
    AugmentConfig("source_balanced_copula", "copula"),
]


def build_source_masks(source_series: pd.Series) -> dict[str, np.ndarray]:
    """Return boolean masks per benign source file."""
    series = source_series.astype(str)
    return {source: (series == source).to_numpy() for source in sorted(series.unique())}


def allocate_equal_target_sizes(source_masks: dict[str, np.ndarray]) -> dict[str, int]:
    """Balance all benign sources to the size of the largest source."""
    counts = {source: int(mask.sum()) for source, mask in source_masks.items()}
    target = max(counts.values())
    return {source: target for source in counts}


def generate_local_interp(x_source: np.ndarray, synthetic_rows: int, seed: int) -> np.ndarray:
    """Generate synthetic rows by interpolating source-local benign neighbours."""
    if synthetic_rows <= 0:
        return np.empty((0, x_source.shape[1]), dtype=float)

    rng = np.random.default_rng(seed)
    knn = NearestNeighbors(n_neighbors=min(6, len(x_source)), n_jobs=-1)
    knn.fit(x_source)
    neighbor_idx = knn.kneighbors(return_distance=False)

    base_idx = rng.integers(0, x_source.shape[0], size=synthetic_rows)
    if neighbor_idx.shape[1] > 1:
        neighbor_slot = rng.integers(1, neighbor_idx.shape[1], size=synthetic_rows)
    else:
        neighbor_slot = np.zeros(shape=(synthetic_rows,), dtype=int)
    partner_idx = neighbor_idx[base_idx, neighbor_slot]

    base = x_source[base_idx]
    partner = x_source[partner_idx]
    lam = rng.beta(0.5, 0.5, size=(synthetic_rows, 1))
    synthetic = base + lam * (partner - base)
    synthetic += rng.normal(loc=0.0, scale=0.01, size=synthetic.shape)

    feature_min = x_source.min(axis=0)
    feature_max = x_source.max(axis=0)
    return np.clip(synthetic, feature_min, feature_max)


def generate_source_gmm(x_source: np.ndarray, synthetic_rows: int, seed: int) -> np.ndarray:
    """Generate synthetic rows from a source-local GMM."""
    if synthetic_rows <= 0:
        return np.empty((0, x_source.shape[1]), dtype=float)

    components = min(16, max(4, len(x_source) // 2000))
    model = GaussianMixture(
        n_components=components,
        covariance_type="tied",
        reg_covar=1e-6,
        random_state=seed,
        max_iter=300,
    )
    model.fit(x_source)
    synthetic, _ = model.sample(synthetic_rows)
    feature_min = x_source.min(axis=0)
    feature_max = x_source.max(axis=0)
    return np.clip(synthetic, feature_min, feature_max)


def generate_source_copula(x_source: np.ndarray, synthetic_rows: int, seed: int) -> np.ndarray:
    """Generate synthetic rows using a Gaussian copula fitted per benign source."""
    if synthetic_rows <= 0:
        return np.empty((0, x_source.shape[1]), dtype=float)

    quantile = QuantileTransformer(
        n_quantiles=min(2000, len(x_source)),
        output_distribution="normal",
        random_state=seed,
        subsample=int(1e9),
    )
    z = quantile.fit_transform(x_source)

    cov = LedoitWolf().fit(z).covariance_
    rng = np.random.default_rng(seed)
    sampled_z = rng.multivariate_normal(mean=np.zeros(z.shape[1]), cov=cov, size=synthetic_rows)
    sampled_x = quantile.inverse_transform(sampled_z)

    feature_min = x_source.min(axis=0)
    feature_max = x_source.max(axis=0)
    return np.clip(sampled_x, feature_min, feature_max)


def augment_by_source(
    x_train: np.ndarray,
    source_series: pd.Series,
    config: AugmentConfig,
) -> tuple[np.ndarray, pd.Series, dict[str, int]]:
    """Balance benign source counts with synthetic rows generated per source."""
    if config.method == "none":
        return (
            x_train.copy(),
            source_series.reset_index(drop=True),
            {str(source): 0 for source in sorted(source_series.unique())},
        )

    source_masks = build_source_masks(source_series)
    targets = allocate_equal_target_sizes(source_masks)
    current_counts = {source: int(mask.sum()) for source, mask in source_masks.items()}

    synthetic_chunks: list[np.ndarray] = []
    synthetic_labels: list[str] = []
    added_counts: dict[str, int] = {}

    for offset, source in enumerate(sorted(source_masks)):
        x_source = x_train[source_masks[source]]
        needed = targets[source] - current_counts[source]
        added_counts[source] = int(needed)
        if needed <= 0:
            continue

        seed = MODEL_RANDOM_SEED + 100 * (offset + 1)
        if config.method == "local_interp":
            synthetic = generate_local_interp(x_source, needed, seed=seed)
        elif config.method == "gmm":
            synthetic = generate_source_gmm(x_source, needed, seed=seed)
        elif config.method == "copula":
            synthetic = generate_source_copula(x_source, needed, seed=seed)
        else:
            raise ValueError(f"Nieznana metoda augmentacji: {config.method}")

        synthetic_chunks.append(synthetic)
        synthetic_labels.extend([source] * len(synthetic))

    if synthetic_chunks:
        synthetic_all = np.vstack(synthetic_chunks)
        x_aug = np.vstack([x_train, synthetic_all])
        source_aug = pd.concat(
            [source_series.reset_index(drop=True), pd.Series(synthetic_labels, dtype=str)], ignore_index=True
        )
    else:
        x_aug = x_train.copy()
        source_aug = source_series.reset_index(drop=True)

    return x_aug, source_aug, added_counts


def evaluate_kmeans_mid_fpr(
    x_train: np.ndarray, x_eval: np.ndarray, eval_frame: pd.DataFrame
) -> tuple[dict[str, int | float | str], np.ndarray]:
    """Evaluate the practical k-means configuration."""
    started = time.perf_counter()
    model = KMeans(n_clusters=1024, random_state=MODEL_RANDOM_SEED, n_init=10)
    model.fit(x_train)
    train_dist = np.linalg.norm(x_train - model.cluster_centers_[model.labels_], axis=1)
    threshold = float(np.percentile(train_dist, 90.0))
    labels = model.predict(x_eval)
    eval_dist = np.linalg.norm(x_eval - model.cluster_centers_[labels], axis=1)
    predictions = (eval_dist > threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)
    row: dict[str, int | float | str] = {
        "algorithm": "kmeans",
        "config_name": "kmeans_mid_fpr",
        "parameters": "k=1024, threshold_percentile=90.0",
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def evaluate_sourceaware_birch(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    source_series: pd.Series,
) -> tuple[dict[str, int | float | str], np.ndarray]:
    """Evaluate the best practical source-aware Birch configuration."""
    started = time.perf_counter()
    source_masks = build_source_masks(source_series)
    train_scores_by_source: list[np.ndarray] = []
    eval_scores_by_source: list[np.ndarray] = []
    cluster_counts: dict[str, int] = {}

    for source in sorted(source_masks):
        x_source = x_train[source_masks[source]]
        model = Birch(threshold=0.75, branching_factor=50, n_clusters=None)
        model.fit(x_source)
        centers = np.asarray(model.subcluster_centers_)
        cluster_counts[source] = int(len(centers))

        local_train = distance_to_centers(x_source, centers)
        local_threshold = float(np.percentile(local_train, 94.0))
        if local_threshold <= 0:
            local_threshold = 1e-9

        train_scores_by_source.append(distance_to_centers(x_train, centers) / local_threshold)
        eval_scores_by_source.append(distance_to_centers(x_eval, centers) / local_threshold)

    train_scores = np.min(np.vstack(train_scores_by_source), axis=0)
    eval_scores = np.min(np.vstack(eval_scores_by_source), axis=0)
    threshold = float(np.percentile(train_scores, 94.0))
    predictions = (eval_scores > threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)

    row: dict[str, int | float | str] = {
        "algorithm": "birch_sourceaware",
        "config_name": "birch_sourceaware_p94",
        "parameters": "threshold=0.75, branching_factor=50, local_percentile=94.0, global_percentile=94.0",
        "cluster_count": int(sum(cluster_counts.values())),
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def main() -> None:
    """Run source-balanced benign augmentation experiments for the main clustering models."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_frame, eval_frame = load_clustering_samples_with_train_size(TRAIN_BENIGN_ROWS)
    x_train, x_eval, used_columns = prepare_features(train_frame, eval_frame)
    source_series = train_frame["source_file"].astype(str).reset_index(drop=True)

    raw_rows: list[dict[str, int | float | str]] = []
    recall_rows: list[dict[str, int | float | str]] = []

    for config in AUGMENT_CONFIGS:
        x_aug, source_aug, added_counts = augment_by_source(x_train, source_series, config)
        source_summary = "; ".join(
            f"{Path(source).name}:+{count}" for source, count in sorted(added_counts.items())
        )

        for algorithm in ("kmeans", "dbscan", "birch_sourceaware"):
            if algorithm == "kmeans":
                metrics, predictions = evaluate_kmeans_mid_fpr(x_aug, x_eval, eval_frame)
            elif algorithm == "dbscan":
                metrics, predictions = evaluate_dbscan(
                    x_aug,
                    x_eval,
                    eval_frame,
                    {
                        "name": "dbscan_refined_eps0.38_m8",
                        "eps": 0.38,
                        "min_samples": 8,
                        "eps_percentile": "manual_local",
                    },
                )
            else:
                metrics, predictions = evaluate_sourceaware_birch(x_aug, x_eval, eval_frame, source_aug)

            metrics["augmentation_name"] = config.name
            metrics["augmentation_method"] = config.method
            metrics["train_rows_total"] = int(x_aug.shape[0])
            metrics["synthetic_rows_total"] = int(x_aug.shape[0] - x_train.shape[0])
            metrics["feature_count"] = len(used_columns)
            metrics["source_balance"] = source_summary
            raw_rows.append(metrics)

            for row in per_class_recall(eval_frame, predictions, str(metrics["config_name"])):
                row["algorithm"] = algorithm
                row["augmentation_name"] = config.name
                row["augmentation_method"] = config.method
                row["train_rows_total"] = int(x_aug.shape[0])
                recall_rows.append(row)

    raw_frame = pd.DataFrame(raw_rows).sort_values(["algorithm", "augmentation_name"])
    recall_frame = pd.DataFrame(recall_rows).sort_values(["algorithm", "augmentation_name", "attack_class"])

    summary = raw_frame[
        [
            "algorithm",
            "config_name",
            "augmentation_name",
            "augmentation_method",
            "train_rows_total",
            "synthetic_rows_total",
            "precision",
            "recall",
            "f1",
            "fpr",
            "runtime_seconds",
            "source_balance",
        ]
    ]

    raw_path = RESULTS_DIR / "python_benign_source_balancing_study.csv"
    recall_path = RESULTS_DIR / "python_benign_source_balancing_recall_by_class.csv"
    summary_path = RESULTS_DIR / "python_benign_source_balancing_summary.csv"

    raw_frame.to_csv(raw_path, index=False)
    recall_frame.to_csv(recall_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Zapisano wyniki source-balanced augmentacji Benign:")
    print(raw_path)
    print(recall_path)
    print(summary_path)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    from pathlib import Path

    main()
