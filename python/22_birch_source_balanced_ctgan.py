from __future__ import annotations

import time

import numpy as np
import pandas as pd
from ctgan import CTGAN
from sklearn.cluster import Birch

from clustering_stage import distance_to_centers, per_class_recall
from data_utils import (
    MODEL_RANDOM_SEED,
    RESULTS_DIR,
    load_clustering_samples_with_train_size,
    metrics_table,
    prepare_features,
)


TRAIN_BENIGN_ROWS = 100_000
GLOBAL_PERCENTILES = [94, 95, 96, 97]
CTGAN_EPOCHS = 40


def build_source_masks(source_series: pd.Series) -> dict[str, np.ndarray]:
    """Return boolean masks per benign source file."""
    series = source_series.astype(str)
    return {source: (series == source).to_numpy() for source in sorted(series.unique())}


def target_size_per_source(source_masks: dict[str, np.ndarray]) -> int:
    """Use the largest benign source as the balancing target."""
    return max(int(mask.sum()) for mask in source_masks.values())


def sample_ctgan_per_source(
    x_train: np.ndarray,
    source_series: pd.Series,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.Series, dict[str, int]]:
    """Fit CTGAN separately on each benign source and balance all sources to the largest one."""
    source_masks = build_source_masks(source_series)
    target_size = target_size_per_source(source_masks)

    chunks: list[np.ndarray] = []
    labels: list[str] = []
    added_counts: dict[str, int] = {}

    for idx, source in enumerate(sorted(source_masks)):
        x_source = x_train[source_masks[source]]
        needed = target_size - len(x_source)
        added_counts[source] = int(needed)
        if needed <= 0:
            continue

        frame = pd.DataFrame(x_source, columns=feature_names)
        seed = MODEL_RANDOM_SEED + 500 * (idx + 1)
        synthesizer = CTGAN(
            embedding_dim=128,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            batch_size=500,
            epochs=CTGAN_EPOCHS,
            verbose=False,
            enable_gpu=False,
        )
        synthesizer.set_random_state(seed)
        synthesizer.fit(frame, discrete_columns=[])
        sampled = synthesizer.sample(needed)
        sampled = sampled[feature_names]
        synthetic = sampled.to_numpy(dtype=float)
        synthetic = np.clip(synthetic, x_source.min(axis=0), x_source.max(axis=0))

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
        "config_name": f"birch_ctgan_{int(global_percentile)}",
        "parameters": (
            f"threshold=0.75, branching_factor=50, local_percentile=94.0, "
            f"global_percentile={global_percentile}, ctgan_epochs={CTGAN_EPOCHS}"
        ),
        "cluster_count": int(sum(cluster_counts.values())),
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def main() -> None:
    """Run per-source CTGAN augmentation for source-aware Birch."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_started = time.perf_counter()
    train_frame, eval_frame = load_clustering_samples_with_train_size(TRAIN_BENIGN_ROWS)
    x_train, x_eval, feature_names = prepare_features(train_frame, eval_frame)
    source_series = train_frame["source_file"].astype(str)

    x_aug, source_aug, added_counts = sample_ctgan_per_source(x_train, source_series, feature_names)
    source_balance = "; ".join(
        f"{source.split('/')[-1]}:+{count}" for source, count in sorted(added_counts.items())
    )

    raw_rows: list[dict[str, int | float | str]] = []
    recall_rows: list[dict[str, int | float | str]] = []

    for global_percentile in GLOBAL_PERCENTILES:
        metrics, predictions = evaluate_birch_sourceaware(
            x_train=x_aug,
            x_eval=x_eval,
            eval_frame=eval_frame,
            source_series=source_aug,
            global_percentile=float(global_percentile),
        )
        metrics["augmentation_name"] = "source_balanced_ctgan"
        metrics["augmentation_method"] = "ctgan"
        metrics["train_rows_total"] = int(x_aug.shape[0])
        metrics["synthetic_rows_total"] = int(x_aug.shape[0] - x_train.shape[0])
        metrics["source_balance"] = source_balance
        metrics["ctgan_epochs"] = CTGAN_EPOCHS
        raw_rows.append(metrics)

        for row in per_class_recall(eval_frame, predictions, str(metrics["config_name"])):
            row["augmentation_name"] = "source_balanced_ctgan"
            row["augmentation_method"] = "ctgan"
            row["train_rows_total"] = int(x_aug.shape[0])
            recall_rows.append(row)

    raw_frame = pd.DataFrame(raw_rows).sort_values(["fpr", "f1"], ascending=[True, False])
    recall_frame = pd.DataFrame(recall_rows).sort_values(["config_name", "attack_class"])

    raw_path = RESULTS_DIR / "python_birch_source_balanced_ctgan.csv"
    recall_path = RESULTS_DIR / "python_birch_source_balanced_ctgan_recall_by_class.csv"

    raw_frame.to_csv(raw_path, index=False)
    recall_frame.to_csv(recall_path, index=False)

    print("Zapisano wyniki source-balanced CTGAN dla Birch source-aware:")
    print(raw_path)
    print(recall_path)
    print()
    print(
        raw_frame[
            [
                "config_name",
                "precision",
                "recall",
                "f1",
                "fpr",
                "runtime_seconds",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Laczny czas eksperymentu: {time.perf_counter() - overall_started:.2f} s")


if __name__ == "__main__":
    main()
