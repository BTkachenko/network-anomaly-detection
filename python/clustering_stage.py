from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import Birch, DBSCAN, OPTICS
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from data_utils import MODEL_RANDOM_SEED, metrics_table


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


def estimate_density_reference(x_train: np.ndarray, neighbor_count: int = 10) -> pd.DataFrame:
    """Estimate a data-driven eps scale from k-nearest-neighbour distances."""
    nn = NearestNeighbors(n_neighbors=neighbor_count, n_jobs=-1)
    nn.fit(x_train)
    distances, _ = nn.kneighbors(x_train)
    kth_distance = distances[:, -1]

    rows = []
    for percentile in (75, 80, 85, 90, 95):
        rows.append(
            {
                "neighbor_count": neighbor_count,
                "percentile": percentile,
                "distance": float(np.percentile(kth_distance, percentile)),
            }
        )
    return pd.DataFrame(rows)


def distance_to_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return distance to the nearest cluster representative."""
    if centers.size == 0:
        return np.full(shape=(x.shape[0],), fill_value=np.inf, dtype=float)
    _, distances = pairwise_distances_argmin_min(x, centers)
    return distances


def evaluate_birch(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Fit Birch on benign train and classify distant points as anomalies."""
    started = time.perf_counter()
    model = Birch(
        threshold=float(config["threshold"]),
        branching_factor=int(config["branching_factor"]),
        n_clusters=None,
    )
    model.fit(x_train)
    centers = np.asarray(model.subcluster_centers_)
    train_scores = distance_to_centers(x_train, centers)
    score_threshold = float(np.percentile(train_scores, float(config["score_percentile"])))
    eval_scores = distance_to_centers(x_eval, centers)
    predictions = (eval_scores > score_threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)

    row: dict[str, float | int | str] = {
        "algorithm": "birch",
        "config_name": str(config["name"]),
        "parameters": (
            f"threshold={config['threshold']}, branching_factor={config['branching_factor']}, "
            f"score_percentile={config['score_percentile']}"
        ),
        "cluster_count": int(len(centers)),
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def evaluate_gmm(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Fit a Gaussian mixture on benign train and flag low-likelihood points."""
    started = time.perf_counter()
    model = GaussianMixture(
        n_components=int(config["n_components"]),
        covariance_type=str(config["covariance_type"]),
        reg_covar=float(config["reg_covar"]),
        random_state=MODEL_RANDOM_SEED,
        max_iter=300,
    )
    model.fit(x_train)
    train_scores = model.score_samples(x_train)
    score_threshold = float(np.percentile(train_scores, float(config["lower_percentile"])))
    eval_scores = model.score_samples(x_eval)
    predictions = (eval_scores < score_threshold).astype(int)
    metrics = metrics_table(eval_frame, predictions)

    row: dict[str, float | int | str] = {
        "algorithm": "gmm",
        "config_name": str(config["name"]),
        "parameters": (
            f"n_components={config['n_components']}, covariance_type={config['covariance_type']}, "
            f"reg_covar={config['reg_covar']}, lower_percentile={config['lower_percentile']}"
        ),
        "cluster_count": int(config["n_components"]),
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def density_core_predictions(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    core_samples: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score samples by distance to density core representatives."""
    train_scores = distance_to_centers(x_train, core_samples)
    eval_scores = distance_to_centers(x_eval, core_samples)
    return train_scores, (eval_scores > threshold).astype(int)


def evaluate_dbscan(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Fit DBSCAN on benign train and classify points outside core neighborhoods as anomalies."""
    started = time.perf_counter()
    eps = float(config["eps"])
    model = DBSCAN(eps=eps, min_samples=int(config["min_samples"]), n_jobs=-1)
    labels = model.fit_predict(x_train)

    if len(model.core_sample_indices_) == 0:
        predictions = np.ones(shape=(x_eval.shape[0],), dtype=int)
        cluster_count = 0
        noise_ratio = 1.0
    else:
        core_samples = x_train[model.core_sample_indices_]
        _, predictions = density_core_predictions(x_train, x_eval, core_samples, eps)
        cluster_count = int(len(set(labels)) - (1 if -1 in labels else 0))
        noise_ratio = float((labels == -1).mean())

    metrics = metrics_table(eval_frame, predictions)
    row: dict[str, float | int | str] = {
        "algorithm": "dbscan",
        "config_name": str(config["name"]),
        "parameters": f"eps={eps:.6f}, min_samples={config['min_samples']}, eps_percentile={config.get('eps_percentile', 'manual')}",
        "cluster_count": cluster_count,
        "train_noise_ratio": noise_ratio,
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def evaluate_optics(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Fit OPTICS on benign train and use core neighborhoods as the normal region."""
    started = time.perf_counter()
    max_eps = float(config["max_eps"])
    model = OPTICS(
        min_samples=int(config["min_samples"]),
        max_eps=max_eps,
        cluster_method="xi",
        n_jobs=-1,
    )
    labels = model.fit_predict(x_train)
    core_mask = np.isfinite(model.core_distances_) & (model.core_distances_ <= max_eps)

    if not np.any(core_mask):
        predictions = np.ones(shape=(x_eval.shape[0],), dtype=int)
        cluster_count = 0
        noise_ratio = 1.0
    else:
        core_samples = x_train[core_mask]
        _, predictions = density_core_predictions(x_train, x_eval, core_samples, max_eps)
        cluster_count = int(len(set(labels)) - (1 if -1 in labels else 0))
        noise_ratio = float((labels == -1).mean())

    metrics = metrics_table(eval_frame, predictions)
    row: dict[str, float | int | str] = {
        "algorithm": "optics",
        "config_name": str(config["name"]),
        "parameters": (
            f"max_eps={max_eps:.6f}, min_samples={config['min_samples']}, "
            f"eps_percentile={config.get('eps_percentile', 'manual')}"
        ),
        "cluster_count": cluster_count,
        "train_noise_ratio": noise_ratio,
        **metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    return row, predictions


def build_birch_configs() -> list[dict[str, float | int | str]]:
    """Return three Birch configurations around conservative to sensitive settings."""
    return [
        {"name": "birch_practical", "threshold": 1.25, "branching_factor": 50, "score_percentile": 95.0},
        {"name": "birch_balanced", "threshold": 1.00, "branching_factor": 50, "score_percentile": 92.0},
        {"name": "birch_higher_recall", "threshold": 0.75, "branching_factor": 50, "score_percentile": 90.0},
    ]


def build_gmm_configs() -> list[dict[str, float | int | str]]:
    """Return three GMM configurations from conservative to sensitive."""
    return [
        {
            "name": "gmm_practical",
            "n_components": 8,
            "covariance_type": "diag",
            "reg_covar": 1e-6,
            "lower_percentile": 1.0,
        },
        {
            "name": "gmm_balanced",
            "n_components": 16,
            "covariance_type": "diag",
            "reg_covar": 1e-6,
            "lower_percentile": 5.0,
        },
        {
            "name": "gmm_higher_recall",
            "n_components": 24,
            "covariance_type": "diag",
            "reg_covar": 1e-6,
            "lower_percentile": 10.0,
        },
    ]


def build_density_configs(reference: pd.DataFrame, kind: str) -> list[dict[str, float | int | str]]:
    """Build DBSCAN or OPTICS configs from data-driven eps candidates."""
    distance_by_percentile = {
        int(row["percentile"]): float(row["distance"]) for _, row in reference.iterrows()
    }
    if kind == "dbscan":
        return [
            {
                "name": "dbscan_practical",
                "eps": distance_by_percentile[80],
                "eps_percentile": 80,
                "min_samples": 10,
            },
            {
                "name": "dbscan_balanced",
                "eps": distance_by_percentile[85],
                "eps_percentile": 85,
                "min_samples": 10,
            },
            {
                "name": "dbscan_higher_recall",
                "eps": distance_by_percentile[90],
                "eps_percentile": 90,
                "min_samples": 5,
            },
        ]
    return [
        {
            "name": "optics_practical",
            "max_eps": distance_by_percentile[80],
            "eps_percentile": 80,
            "min_samples": 10,
        },
        {
            "name": "optics_balanced",
            "max_eps": distance_by_percentile[85],
            "eps_percentile": 85,
            "min_samples": 10,
        },
        {
            "name": "optics_higher_recall",
            "max_eps": distance_by_percentile[90],
            "eps_percentile": 90,
            "min_samples": 5,
        },
    ]


def run_algorithm_config(
    algorithm: str,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Dispatch evaluation to the selected clustering algorithm."""
    if algorithm == "birch":
        return evaluate_birch(x_train, x_eval, eval_frame, config)
    if algorithm == "gmm":
        return evaluate_gmm(x_train, x_eval, eval_frame, config)
    if algorithm == "dbscan":
        return evaluate_dbscan(x_train, x_eval, eval_frame, config)
    if algorithm == "optics":
        return evaluate_optics(x_train, x_eval, eval_frame, config)
    raise ValueError(f"Nieznany algorytm: {algorithm}")


def summarize_grid(
    frame: pd.DataFrame,
    algorithm: str,
    fpr_limits: tuple[float, ...] = (0.10, 0.15, 0.20),
) -> pd.DataFrame:
    """Return a short summary with the best config overall and under FPR limits."""
    subset = frame.loc[frame["algorithm"] == algorithm].copy()
    rows: list[dict[str, str | float]] = []

    if subset.empty:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "criterion",
                "config_name",
                "precision",
                "recall",
                "f1",
                "fpr",
                "parameters",
            ]
        )

    best_f1 = subset.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    rows.append(
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

    for limit in fpr_limits:
        limited = subset.loc[subset["fpr"] <= limit]
        if limited.empty:
            continue
        best = limited.sort_values(["recall", "f1", "precision"], ascending=[False, False, False]).iloc[0]
        rows.append(
            {
                "algorithm": algorithm,
                "criterion": f"best_recall_fpr_le_{limit:.2f}",
                "config_name": best["config_name"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "fpr": best["fpr"],
                "parameters": best["parameters"],
            }
        )

    return pd.DataFrame(rows)
