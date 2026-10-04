from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer

from clustering_stage import distance_to_centers, per_class_recall
from data_utils import (
    MODEL_RANDOM_SEED,
    RESULTS_DIR,
    load_clustering_samples_with_train_size,
    metrics_table,
    prepare_features,
)


TRAIN_BENIGN_ROWS = 100_000
GLOBAL_PERCENTILES = [94, 95, 96, 97, 98]


@dataclass(frozen=True)
class GeneratorConfig:
    name: str
    method: str


GENERATOR_CONFIGS = [
    GeneratorConfig("source_balanced_gmm", "gmm"),
    GeneratorConfig("source_balanced_copula", "copula"),
]


def build_source_masks(source_series: pd.Series) -> dict[str, np.ndarray]:
    """Return boolean masks per benign source file."""
    series = source_series.astype(str)
    return {source: (series == source).to_numpy() for source in sorted(series.unique())}


def source_target_counts(source_masks: dict[str, np.ndarray]) -> tuple[dict[str, int], int]:
    """Balance all sources to the size of the largest source."""
    counts = {source: int(mask.sum()) for source, mask in source_masks.items()}
    target = max(counts.values())
    return counts, target


def generate_source_gmm(x_source: np.ndarray, synthetic_rows: int, seed: int) -> np.ndarray:
    """Generate synthetic benign rows per source with a tied-covariance GMM."""
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
    return np.clip(synthetic, x_source.min(axis=0), x_source.max(axis=0))


def generate_source_copula(x_source: np.ndarray, synthetic_rows: int, seed: int) -> np.ndarray:
    """Generate synthetic benign rows per source with a Gaussian copula."""
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
    return np.clip(sampled_x, x_source.min(axis=0), x_source.max(axis=0))


def augment_balanced_sources(
    x_train: np.ndarray,
    source_series: pd.Series,
    generator: GeneratorConfig,
) -> tuple[np.ndarray, pd.Series, dict[str, int]]:
    """Augment all smaller benign sources up to the largest source size."""
    masks = build_source_masks(source_series)
    counts, target = source_target_counts(masks)

    chunks: list[np.ndarray] = []
    labels: list[str] = []
    added_counts: dict[str, int] = {}

    for idx, source in enumerate(sorted(masks)):
        x_source = x_train[masks[source]]
        needed = target - counts[source]
        added_counts[source] = int(needed)
        if needed <= 0:
            continue

        seed = MODEL_RANDOM_SEED + 200 * (idx + 1)
        if generator.method == "gmm":
            synthetic = generate_source_gmm(x_source, needed, seed)
        elif generator.method == "copula":
            synthetic = generate_source_copula(x_source, needed, seed)
        else:
            raise ValueError(f"Nieznana metoda generatora: {generator.method}")

        chunks.append(synthetic)
        labels.extend([source] * len(synthetic))

    x_aug = np.vstack([x_train] + chunks)
    source_aug = pd.concat(
        [source_series.reset_index(drop=True), pd.Series(labels, dtype=str)], ignore_index=True
    )
    return x_aug, source_aug, added_counts


def evaluate_birch_sourceaware(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    source_series: pd.Series,
    global_percentile: float,
) -> tuple[dict[str, int | float | str], np.ndarray]:
    """Evaluate source-aware Birch for a chosen global percentile."""
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
    threshold = float(np.percentile(train_scores, global_percentile))
    predictions = (eval_scores > threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)

    row: dict[str, int | float | str] = {
        "algorithm": "birch_sourceaware",
        "config_name": f"birch_{int(global_percentile)}",
        "parameters": f"threshold=0.75, branching_factor=50, local_percentile=94.0, global_percentile={global_percentile}",
        "cluster_count": int(sum(cluster_counts.values())),
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def main() -> None:
    """Run a threshold sweep for source-balanced Birch variants."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_frame, eval_frame = load_clustering_samples_with_train_size(TRAIN_BENIGN_ROWS)
    x_train, x_eval, _ = prepare_features(train_frame, eval_frame)
    source_series = train_frame["source_file"].astype(str)

    raw_rows: list[dict[str, int | float | str]] = []
    recall_rows: list[dict[str, int | float | str]] = []

    for generator in GENERATOR_CONFIGS:
        x_aug, source_aug, added_counts = augment_balanced_sources(x_train, source_series, generator)
        source_balance = "; ".join(
            f"{source.split('/')[-1]}:+{count}" for source, count in sorted(added_counts.items())
        )

        for global_percentile in GLOBAL_PERCENTILES:
            metrics, predictions = evaluate_birch_sourceaware(
                x_aug,
                x_eval,
                eval_frame,
                source_aug,
                global_percentile=float(global_percentile),
            )
            metrics["augmentation_name"] = generator.name
            metrics["augmentation_method"] = generator.method
            metrics["train_rows_total"] = int(x_aug.shape[0])
            metrics["synthetic_rows_total"] = int(x_aug.shape[0] - x_train.shape[0])
            metrics["source_balance"] = source_balance
            raw_rows.append(metrics)

            for row in per_class_recall(eval_frame, predictions, str(metrics["config_name"])):
                row["augmentation_name"] = generator.name
                row["augmentation_method"] = generator.method
                row["train_rows_total"] = int(x_aug.shape[0])
                recall_rows.append(row)

    raw_frame = pd.DataFrame(raw_rows).sort_values(
        ["augmentation_name", "fpr", "f1"], ascending=[True, True, False]
    )
    recall_frame = pd.DataFrame(recall_rows).sort_values(["augmentation_name", "config_name", "attack_class"])

    raw_path = RESULTS_DIR / "python_birch_source_balanced_sweep.csv"
    recall_path = RESULTS_DIR / "python_birch_source_balanced_sweep_recall_by_class.csv"

    raw_frame.to_csv(raw_path, index=False)
    recall_frame.to_csv(recall_path, index=False)

    print("Zapisano sweep progów dla source-balanced Birch:")
    print(raw_path)
    print(recall_path)
    print()
    print(
        raw_frame[
            [
                "augmentation_name",
                "config_name",
                "precision",
                "recall",
                "f1",
                "fpr",
                "runtime_seconds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
