from __future__ import annotations

import time

import hdbscan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data_utils import metrics_table


HDBSCAN_PCA_COMPONENTS = 24

HDBSCAN_FINAL_CONFIGS = [
    {"name": "hdbscan_fine_structure", "min_cluster_size": 100, "min_samples": 10},
    {"name": "hdbscan_balanced_structure", "min_cluster_size": 250, "min_samples": 10},
    {"name": "hdbscan_conservative_structure", "min_cluster_size": 500, "min_samples": 25},
]

HDBSCAN_CLUSTER_SIZE_TREND = [
    {"setting": "min_samples=10", "min_cluster_size": value, "min_samples": 10}
    for value in (100, 250, 500, 1000)
]

HDBSCAN_MIN_SAMPLES_TREND = [
    {"setting": "min_cluster_size=250", "min_cluster_size": 250, "min_samples": value}
    for value in (5, 10, 25, 50)
]


def build_hdbscan_space(x_eval: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Build the lower-dimensional space used by HDBSCAN.

    HDBSCAN in the full 72-dimensional standardized space is both slower and
    less interpretable. A PCA projection keeps almost all variance while making
    density clustering practical on the mixed sample.
    """

    projector = PCA(n_components=HDBSCAN_PCA_COMPONENTS, svd_solver="full", random_state=2029)
    x_projected = projector.fit_transform(x_eval)
    explained_variance = float(projector.explained_variance_ratio_.sum())
    return x_projected, explained_variance


def evaluate_hdbscan(
    x_projected: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, object],
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """
    Run one HDBSCAN configuration on the mixed sample.

    The main output of this stage is structural: cluster labels, probabilities,
    outlier scores and noise ratios. For orientation only, the noise label (-1)
    is also treated as an anomaly alarm and converted into precision/recall/F1/FPR.
    """

    started = time.perf_counter()
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(config["min_cluster_size"]),
        min_samples=int(config["min_samples"]),
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    model.fit(x_projected)

    labels = model.labels_.astype(int)
    probabilities = model.probabilities_.astype(float)
    outlier_scores = model.outlier_scores_.astype(float)
    noise_predictions = (labels == -1).astype(int)
    metrics = metrics_table(eval_frame, noise_predictions)

    benign_mask = eval_frame["binary_label"].astype(int).to_numpy() == 0
    attack_mask = ~benign_mask
    clusters_found = int(len(set(labels)) - (1 if -1 in labels else 0))

    summary = {
        "config_name": str(config["name"]),
        "min_cluster_size": int(config["min_cluster_size"]),
        "min_samples": int(config["min_samples"]),
        "clusters_found": clusters_found,
        "noise_ratio": float((labels == -1).mean()),
        "benign_noise_ratio": float((labels[benign_mask] == -1).mean()),
        "attack_noise_ratio": float((labels[attack_mask] == -1).mean()),
        "mean_probability": float(probabilities.mean()),
        "mean_outlier_score": float(outlier_scores.mean()),
        "noise_precision": metrics["precision"],
        "noise_recall": metrics["recall"],
        "noise_f1": metrics["f1"],
        "noise_fpr": metrics["fpr"],
        "runtime_seconds": time.perf_counter() - started,
    }

    assignments = eval_frame.loc[
        :, ["coarse_label", "fine_label", "binary_label", "source_file", "source_row"]
    ].copy()
    assignments.insert(0, "config_name", str(config["name"]))
    assignments["cluster_label"] = labels
    assignments["membership_probability"] = probabilities
    assignments["outlier_score"] = outlier_scores
    assignments["noise_prediction"] = noise_predictions
    return summary, assignments


def summarize_by_class(assignments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate HDBSCAN outputs separately for each attack family and benign traffic."""

    rows: list[dict[str, object]] = []

    for (config_name, coarse_label), frame in assignments.groupby(["config_name", "coarse_label"], sort=True):
        noise_ratio = float((frame["cluster_label"] == -1).mean())
        non_noise = frame.loc[frame["cluster_label"] != -1]
        touched_clusters = int(non_noise["cluster_label"].nunique()) if not non_noise.empty else 0

        dominant_cluster_share = 0.0
        if not non_noise.empty:
            dominant_cluster_share = float(non_noise["cluster_label"].value_counts(normalize=True).iloc[0])

        rows.append(
            {
                "config_name": config_name,
                "coarse_label": coarse_label,
                "rows": int(len(frame)),
                "noise_ratio": noise_ratio,
                "mean_probability": float(frame["membership_probability"].mean()),
                "mean_outlier_score": float(frame["outlier_score"].mean()),
                "touched_clusters": touched_clusters,
                "dominant_cluster_share": dominant_cluster_share,
            }
        )

    return pd.DataFrame(rows)


def summarize_by_cluster(assignments: pd.DataFrame) -> pd.DataFrame:
    """Describe the composition of each cluster, including the noise label."""

    rows: list[dict[str, object]] = []

    for (config_name, cluster_label), frame in assignments.groupby(
        ["config_name", "cluster_label"], sort=True
    ):
        label_counts = frame["coarse_label"].value_counts()
        dominant_label = str(label_counts.index[0])
        dominant_share = float(label_counts.iloc[0] / len(frame))
        benign_rows = int((frame["binary_label"].astype(int) == 0).sum())
        attack_rows = int((frame["binary_label"].astype(int) == 1).sum())

        rows.append(
            {
                "config_name": config_name,
                "cluster_label": int(cluster_label),
                "rows": int(len(frame)),
                "benign_rows": benign_rows,
                "attack_rows": attack_rows,
                "benign_share": float(benign_rows / len(frame)),
                "attack_share": float(attack_rows / len(frame)),
                "dominant_label": dominant_label,
                "dominant_label_share": dominant_share,
                "mean_probability": float(frame["membership_probability"].mean()),
                "mean_outlier_score": float(frame["outlier_score"].mean()),
            }
        )

    return pd.DataFrame(rows)


def run_final_configs(x_projected: np.ndarray, eval_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the selected HDBSCAN configurations and collect per-row assignments."""

    summary_rows: list[dict[str, object]] = []
    assignment_frames: list[pd.DataFrame] = []

    for config in HDBSCAN_FINAL_CONFIGS:
        summary, assignments = evaluate_hdbscan(x_projected, eval_frame, config)
        summary_rows.append(summary)
        assignment_frames.append(assignments)

    return pd.DataFrame(summary_rows), pd.concat(assignment_frames, ignore_index=True)


def run_trends(x_projected: np.ndarray, eval_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the small HDBSCAN parameter sweeps used in the report."""

    cluster_size_rows: list[dict[str, object]] = []
    min_samples_rows: list[dict[str, object]] = []

    for config in HDBSCAN_CLUSTER_SIZE_TREND:
        summary, _ = evaluate_hdbscan(
            x_projected,
            eval_frame,
            {
                "name": f"hdbscan_cluster_size_{config['min_cluster_size']}",
                "min_cluster_size": config["min_cluster_size"],
                "min_samples": config["min_samples"],
            },
        )
        cluster_size_rows.append(
            {
                "setting": config["setting"],
                "min_cluster_size": config["min_cluster_size"],
                "clusters_found": summary["clusters_found"],
                "noise_ratio": summary["noise_ratio"],
                "mean_probability": summary["mean_probability"],
                "mean_outlier_score": summary["mean_outlier_score"],
                "noise_precision": summary["noise_precision"],
                "noise_recall": summary["noise_recall"],
                "noise_f1": summary["noise_f1"],
                "noise_fpr": summary["noise_fpr"],
                "runtime_seconds": summary["runtime_seconds"],
            }
        )

    for config in HDBSCAN_MIN_SAMPLES_TREND:
        summary, _ = evaluate_hdbscan(
            x_projected,
            eval_frame,
            {
                "name": f"hdbscan_min_samples_{config['min_samples']}",
                "min_cluster_size": config["min_cluster_size"],
                "min_samples": config["min_samples"],
            },
        )
        min_samples_rows.append(
            {
                "setting": config["setting"],
                "min_samples": config["min_samples"],
                "clusters_found": summary["clusters_found"],
                "noise_ratio": summary["noise_ratio"],
                "mean_probability": summary["mean_probability"],
                "mean_outlier_score": summary["mean_outlier_score"],
                "noise_precision": summary["noise_precision"],
                "noise_recall": summary["noise_recall"],
                "noise_f1": summary["noise_f1"],
                "noise_fpr": summary["noise_fpr"],
                "runtime_seconds": summary["runtime_seconds"],
            }
        )

    return pd.DataFrame(cluster_size_rows), pd.DataFrame(min_samples_rows)
