from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import Birch, DBSCAN, KMeans

from clustering_stage import distance_to_centers, per_class_recall
from data_utils import (
    MODEL_RANDOM_SEED,
    RESULTS_DIR,
    load_clustering_samples_with_train_size,
    metrics_table,
    prepare_features,
)


TRAIN_BENIGN_ROWS = 100_000


@dataclass(frozen=True)
class KMeansSourceConfig:
    name: str
    total_k: int
    local_percentile: float
    global_percentile: float


@dataclass(frozen=True)
class BirchSourceConfig:
    name: str
    threshold: float
    branching_factor: int
    local_percentile: float
    global_percentile: float


@dataclass(frozen=True)
class DbscanSourceConfig:
    name: str
    eps: float
    min_samples: int
    global_percentile: float


KMEANS_SOURCE_CONFIGS = [
    KMeansSourceConfig("kmeans_sourceaware_p90", total_k=1024, local_percentile=90.0, global_percentile=90.0),
    KMeansSourceConfig("kmeans_sourceaware_p91", total_k=1024, local_percentile=90.0, global_percentile=91.0),
    KMeansSourceConfig("kmeans_sourceaware_p92", total_k=1024, local_percentile=90.0, global_percentile=92.0),
]

BIRCH_SOURCE_CONFIGS = [
    BirchSourceConfig(
        "birch_sourceaware_p93",
        threshold=0.75,
        branching_factor=50,
        local_percentile=94.0,
        global_percentile=93.0,
    ),
    BirchSourceConfig(
        "birch_sourceaware_p94",
        threshold=0.75,
        branching_factor=50,
        local_percentile=94.0,
        global_percentile=94.0,
    ),
    BirchSourceConfig(
        "birch_sourceaware_p95",
        threshold=0.75,
        branching_factor=50,
        local_percentile=94.0,
        global_percentile=95.0,
    ),
]

DBSCAN_SOURCE_CONFIGS = [
    DbscanSourceConfig("dbscan_sourceaware_p90", eps=0.38, min_samples=8, global_percentile=90.0),
    DbscanSourceConfig("dbscan_sourceaware_p92", eps=0.38, min_samples=8, global_percentile=92.0),
    DbscanSourceConfig("dbscan_sourceaware_p94", eps=0.38, min_samples=8, global_percentile=94.0),
]


def allocate_k_per_source(source_counts: pd.Series, total_k: int) -> dict[str, int]:
    """Split the total k across benign source files proportionally to sample size."""
    weights = source_counts / source_counts.sum()
    raw = weights * total_k
    base = np.floor(raw).astype(int)
    remainder = total_k - int(base.sum())

    fractions = (raw - base).sort_values(ascending=False)
    for label in fractions.index[:remainder]:
        base[label] += 1

    allocation = {str(label): max(8, int(value)) for label, value in base.items()}
    adjusted_total = sum(allocation.values())
    if adjusted_total != total_k:
        largest = max(allocation, key=allocation.get)
        allocation[largest] += total_k - adjusted_total
    return allocation


def build_source_masks(train_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return boolean masks for each benign source file."""
    source_series = train_frame["source_file"].astype(str)
    return {source: (source_series == source).to_numpy() for source in sorted(source_series.unique())}


def sourceaware_kmeans_scores(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    source_masks: dict[str, np.ndarray],
    total_k: int,
    local_percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Score samples against separate k-means models fitted per benign source file."""
    source_counts = pd.Series({source: int(mask.sum()) for source, mask in source_masks.items()})
    k_per_source = allocate_k_per_source(source_counts, total_k=total_k)

    train_scores_by_source: list[np.ndarray] = []
    eval_scores_by_source: list[np.ndarray] = []
    for source, mask in source_masks.items():
        x_sub = x_train[mask]
        model = KMeans(n_clusters=int(k_per_source[source]), random_state=MODEL_RANDOM_SEED, n_init=10)
        model.fit(x_sub)

        local_train = distance_to_centers(x_sub, model.cluster_centers_)
        local_threshold = float(np.percentile(local_train, local_percentile))
        if local_threshold <= 0:
            local_threshold = 1e-9

        train_scores_by_source.append(distance_to_centers(x_train, model.cluster_centers_) / local_threshold)
        eval_scores_by_source.append(distance_to_centers(x_eval, model.cluster_centers_) / local_threshold)

    train_scores = np.min(np.vstack(train_scores_by_source), axis=0)
    eval_scores = np.min(np.vstack(eval_scores_by_source), axis=0)
    return train_scores, eval_scores, k_per_source


def sourceaware_birch_scores(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    source_masks: dict[str, np.ndarray],
    threshold: float,
    branching_factor: int,
    local_percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Score samples against separate Birch models fitted per benign source file."""
    cluster_counts: dict[str, int] = {}
    train_scores_by_source: list[np.ndarray] = []
    eval_scores_by_source: list[np.ndarray] = []

    for source, mask in source_masks.items():
        x_sub = x_train[mask]
        model = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None)
        model.fit(x_sub)
        centers = np.asarray(model.subcluster_centers_)
        cluster_counts[source] = int(len(centers))

        local_train = distance_to_centers(x_sub, centers)
        local_threshold = float(np.percentile(local_train, local_percentile))
        if local_threshold <= 0:
            local_threshold = 1e-9

        train_scores_by_source.append(distance_to_centers(x_train, centers) / local_threshold)
        eval_scores_by_source.append(distance_to_centers(x_eval, centers) / local_threshold)

    train_scores = np.min(np.vstack(train_scores_by_source), axis=0)
    eval_scores = np.min(np.vstack(eval_scores_by_source), axis=0)
    return train_scores, eval_scores, cluster_counts


def sourceaware_dbscan_scores(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    source_masks: dict[str, np.ndarray],
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Score samples against separate DBSCAN core regions fitted per benign source file."""
    cluster_counts: dict[str, int] = {}
    train_scores_by_source: list[np.ndarray] = []
    eval_scores_by_source: list[np.ndarray] = []

    for source, mask in source_masks.items():
        x_sub = x_train[mask]
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(x_sub)
        cluster_counts[source] = int(len(set(labels)) - (1 if -1 in labels else 0))

        if len(model.core_sample_indices_) == 0:
            train_scores_by_source.append(np.full(x_train.shape[0], np.inf, dtype=float))
            eval_scores_by_source.append(np.full(x_eval.shape[0], np.inf, dtype=float))
            continue

        core_samples = x_sub[model.core_sample_indices_]
        train_scores_by_source.append(distance_to_centers(x_train, core_samples) / eps)
        eval_scores_by_source.append(distance_to_centers(x_eval, core_samples) / eps)

    train_scores = np.min(np.vstack(train_scores_by_source), axis=0)
    eval_scores = np.min(np.vstack(eval_scores_by_source), axis=0)
    return train_scores, eval_scores, cluster_counts


def evaluate_from_scores(
    algorithm: str,
    config_name: str,
    parameters: str,
    train_scores: np.ndarray,
    eval_scores: np.ndarray,
    eval_frame: pd.DataFrame,
    started: float,
    extra_fields: dict[str, int | float | str] | None = None,
) -> tuple[dict[str, int | float | str], np.ndarray]:
    """Turn source-aware continuous scores into predictions using a train percentile threshold."""
    threshold = float(parameters.split("global_percentile=")[-1])
    score_threshold = float(np.percentile(train_scores, threshold))
    predictions = (eval_scores > score_threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)
    row: dict[str, int | float | str] = {
        "algorithm": algorithm,
        "config_name": config_name,
        "parameters": parameters,
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    if extra_fields:
        row.update(extra_fields)
    return row, predictions


def main() -> None:
    """Evaluate source-aware clustering models that fit normal traffic per benign source file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_frame, eval_frame = load_clustering_samples_with_train_size(TRAIN_BENIGN_ROWS)
    x_train, x_eval, used_columns = prepare_features(train_frame, eval_frame)
    source_masks = build_source_masks(train_frame)

    raw_rows: list[dict[str, int | float | str]] = []
    recall_rows: list[dict[str, int | float | str]] = []

    for config in KMEANS_SOURCE_CONFIGS:
        started = time.perf_counter()
        train_scores, eval_scores, k_per_source = sourceaware_kmeans_scores(
            x_train=x_train,
            x_eval=x_eval,
            source_masks=source_masks,
            total_k=config.total_k,
            local_percentile=config.local_percentile,
        )
        row, predictions = evaluate_from_scores(
            algorithm="kmeans_sourceaware",
            config_name=config.name,
            parameters=(
                f"total_k={config.total_k}, local_percentile={config.local_percentile}, "
                f"global_percentile={config.global_percentile}"
            ),
            train_scores=train_scores,
            eval_scores=eval_scores,
            eval_frame=eval_frame,
            started=started,
            extra_fields={
                "source_models": len(k_per_source),
                "feature_count": len(used_columns),
                "allocation": "; ".join(f"{source.split('/')[-1]}={k}" for source, k in k_per_source.items()),
            },
        )
        raw_rows.append(row)
        for recall_row in per_class_recall(eval_frame, predictions, config.name):
            recall_row["algorithm"] = "kmeans_sourceaware"
            recall_rows.append(recall_row)

    for config in BIRCH_SOURCE_CONFIGS:
        started = time.perf_counter()
        train_scores, eval_scores, cluster_counts = sourceaware_birch_scores(
            x_train=x_train,
            x_eval=x_eval,
            source_masks=source_masks,
            threshold=config.threshold,
            branching_factor=config.branching_factor,
            local_percentile=config.local_percentile,
        )
        row, predictions = evaluate_from_scores(
            algorithm="birch_sourceaware",
            config_name=config.name,
            parameters=(
                f"threshold={config.threshold}, branching_factor={config.branching_factor}, "
                f"local_percentile={config.local_percentile}, global_percentile={config.global_percentile}"
            ),
            train_scores=train_scores,
            eval_scores=eval_scores,
            eval_frame=eval_frame,
            started=started,
            extra_fields={
                "source_models": len(cluster_counts),
                "cluster_count": int(sum(cluster_counts.values())),
                "feature_count": len(used_columns),
                "allocation": "; ".join(
                    f"{source.split('/')[-1]}={count}" for source, count in cluster_counts.items()
                ),
            },
        )
        raw_rows.append(row)
        for recall_row in per_class_recall(eval_frame, predictions, config.name):
            recall_row["algorithm"] = "birch_sourceaware"
            recall_rows.append(recall_row)

    for config in DBSCAN_SOURCE_CONFIGS:
        started = time.perf_counter()
        train_scores, eval_scores, cluster_counts = sourceaware_dbscan_scores(
            x_train=x_train,
            x_eval=x_eval,
            source_masks=source_masks,
            eps=config.eps,
            min_samples=config.min_samples,
        )
        row, predictions = evaluate_from_scores(
            algorithm="dbscan_sourceaware",
            config_name=config.name,
            parameters=(
                f"eps={config.eps}, min_samples={config.min_samples}, global_percentile={config.global_percentile}"
            ),
            train_scores=train_scores,
            eval_scores=eval_scores,
            eval_frame=eval_frame,
            started=started,
            extra_fields={
                "source_models": len(cluster_counts),
                "cluster_count": int(sum(cluster_counts.values())),
                "feature_count": len(used_columns),
                "allocation": "; ".join(
                    f"{source.split('/')[-1]}={count}" for source, count in cluster_counts.items()
                ),
            },
        )
        raw_rows.append(row)
        for recall_row in per_class_recall(eval_frame, predictions, config.name):
            recall_row["algorithm"] = "dbscan_sourceaware"
            recall_rows.append(recall_row)

    raw_frame = pd.DataFrame(raw_rows).sort_values(["algorithm", "fpr", "f1"], ascending=[True, True, False])
    recall_frame = pd.DataFrame(recall_rows).sort_values(["algorithm", "config_name", "attack_class"])

    summary_rows: list[dict[str, int | float | str]] = []
    for algorithm, group in raw_frame.groupby("algorithm", sort=True):
        best_f1 = group.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
        summary_rows.append(
            {
                "algorithm": algorithm,
                "criterion": "max_f1",
                "config_name": best_f1["config_name"],
                "precision": best_f1["precision"],
                "recall": best_f1["recall"],
                "f1": best_f1["f1"],
                "fpr": best_f1["fpr"],
                "parameters": best_f1["parameters"],
            }
        )
        constrained = group.loc[group["fpr"] <= 0.10]
        if not constrained.empty:
            best_constrained = constrained.sort_values(
                ["recall", "f1", "precision"], ascending=[False, False, False]
            ).iloc[0]
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "criterion": "best_recall_fpr_le_0.10",
                    "config_name": best_constrained["config_name"],
                    "precision": best_constrained["precision"],
                    "recall": best_constrained["recall"],
                    "f1": best_constrained["f1"],
                    "fpr": best_constrained["fpr"],
                    "parameters": best_constrained["parameters"],
                }
            )

    summary_frame = pd.DataFrame(summary_rows)

    raw_path = RESULTS_DIR / "python_source_aware_clustering.csv"
    recall_path = RESULTS_DIR / "python_source_aware_clustering_recall_by_class.csv"
    summary_path = RESULTS_DIR / "python_source_aware_clustering_summary.csv"

    raw_frame.to_csv(raw_path, index=False)
    recall_frame.to_csv(recall_path, index=False)
    summary_frame.to_csv(summary_path, index=False)

    print("Zapisano wyniki source-aware clustering:")
    print(raw_path)
    print(recall_path)
    print(summary_path)
    print()
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
