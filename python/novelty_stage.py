from __future__ import annotations

import time

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from data_utils import MODEL_RANDOM_SEED, metrics_table
from simple_iforest import SimpleIsolationForest


IFOREST_NOVELTY_CONFIGS = [
    {"name": "iforest_practical", "trees": 100, "sample_size": 256, "threshold_percentile": 95.0},
    {"name": "iforest_balanced", "trees": 200, "sample_size": 512, "threshold_percentile": 92.0},
    {"name": "iforest_higher_recall", "trees": 300, "sample_size": 512, "threshold_percentile": 88.0},
]

OCSVM_CONFIGS = [
    {"name": "ocsvm_practical", "nu": 0.08, "gamma": 0.10},
    {"name": "ocsvm_balanced", "nu": 0.10, "gamma": 0.10},
    {"name": "ocsvm_higher_recall", "nu": 0.12, "gamma": 0.10},
]

PCA_CONFIGS = [
    {"name": "pca_practical", "n_components": 28, "threshold_percentile": 90.0},
    {"name": "pca_balanced", "n_components": 24, "threshold_percentile": 88.0},
    {"name": "pca_higher_recall", "n_components": 26, "threshold_percentile": 86.0},
]

ECOD_CONFIGS = [
    {"name": "ecod_practical", "threshold_percentile": 90.0},
    {"name": "ecod_balanced", "threshold_percentile": 84.0},
    {"name": "ecod_higher_recall", "threshold_percentile": 80.0},
]

IFOREST_THRESHOLD_TREND = [
    {
        "algorithm": "isolation_forest",
        "setting": "trees=300,sample=512",
        "trees": 300,
        "sample_size": 512,
        "threshold_percentile": value,
    }
    for value in (95.0, 92.0, 90.0, 88.0, 86.0, 84.0)
]

OCSVM_NU_TREND = [
    {"algorithm": "one_class_svm", "setting": "gamma=0.02", "nu": value, "gamma": 0.02}
    for value in (0.02, 0.05, 0.10, 0.15, 0.20)
]

OCSVM_GAMMA_TREND = [
    {"algorithm": "one_class_svm", "setting": "nu=0.10", "nu": 0.10, "gamma": value}
    for value in (0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.10)
]

OCSVM_REFINED_GRID = [
    {
        "algorithm": "one_class_svm",
        "setting": "local_grid",
        "nu": nu,
        "gamma": gamma,
    }
    for nu in (0.08, 0.10, 0.12, 0.15)
    for gamma in (0.02, 0.03, 0.05, 0.07, 0.10)
]

PCA_COMPONENT_TREND = [
    {
        "algorithm": "pca_reconstruction",
        "setting": "threshold=90",
        "n_components": value,
        "threshold_percentile": 90.0,
    }
    for value in (4, 8, 12, 16, 24, 32)
]

PCA_THRESHOLD_TREND = [
    {
        "algorithm": "pca_reconstruction",
        "setting": "components=24",
        "n_components": 24,
        "threshold_percentile": value,
    }
    for value in (95.0, 92.0, 90.0, 88.0, 86.0, 84.0)
]

PCA_REFINED_GRID = [
    {
        "algorithm": "pca_reconstruction",
        "setting": "local_grid",
        "n_components": n_components,
        "threshold_percentile": threshold,
    }
    for n_components in (24, 26, 28, 30, 32)
    for threshold in (91.0, 90.0, 89.0, 88.0, 87.0, 86.0)
]

ECOD_THRESHOLD_TREND = [
    {
        "algorithm": "ecod",
        "setting": "default_ecod",
        "threshold_percentile": value,
    }
    for value in (98.0, 95.0, 92.0, 90.0, 88.0, 86.0, 84.0, 80.0)
]


def per_class_recall(
    eval_frame: pd.DataFrame, predictions: np.ndarray, config_name: str
) -> list[dict[str, str | float]]:
    """Compute recall separately for every attack family present in the evaluation set."""
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


def evaluate_iforest(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Evaluate one Isolation Forest novelty configuration."""
    started = time.perf_counter()
    model = SimpleIsolationForest(
        n_trees=int(config["trees"]),
        sample_size=int(config["sample_size"]),
        random_state=random_seed,
    )
    model.fit(x_train)
    train_scores = model.score_samples(x_train)
    threshold = float(np.percentile(train_scores, float(config["threshold_percentile"])))
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


def evaluate_ocsvm(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | str],
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[dict[str, float | int], np.ndarray]:
    """
    Evaluate one One-Class SVM configuration.

    The model itself is deterministic for a fixed training matrix and parameters.
    The random_seed argument is kept only for API symmetry with the other models.
    """

    _ = random_seed
    started = time.perf_counter()
    model = OneClassSVM(kernel="rbf", nu=float(config["nu"]), gamma=float(config["gamma"]))
    model.fit(x_train)
    predictions = (model.predict(x_eval) == -1).astype(int)
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


def evaluate_pca_reconstruction(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Evaluate PCA reconstruction error as a novelty detector."""

    started = time.perf_counter()
    model = PCA(
        n_components=int(config["n_components"]),
        svd_solver="full",
        random_state=random_seed,
    )
    model.fit(x_train)

    train_recon = model.inverse_transform(model.transform(x_train))
    eval_recon = model.inverse_transform(model.transform(x_eval))

    train_errors = np.mean((x_train - train_recon) ** 2, axis=1)
    eval_errors = np.mean((x_eval - eval_recon) ** 2, axis=1)

    threshold = float(np.percentile(train_errors, float(config["threshold_percentile"])))
    predictions = (eval_errors > threshold).astype(int)
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


def evaluate_ecod(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    config: dict[str, float | int | str],
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[dict[str, float | int], np.ndarray]:
    """
    Evaluate ECOD as a novelty detector.

    The model is deterministic for a fixed matrix. The random seed is kept only
    for API symmetry with the other models.
    """

    _ = random_seed
    started = time.perf_counter()
    model = ECOD()
    model.fit(x_train)
    train_scores = model.decision_scores_
    eval_scores = model.decision_function(x_eval)

    threshold = float(np.percentile(train_scores, float(config["threshold_percentile"])))
    predictions = (eval_scores > threshold).astype(int)
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
    """Run the novelty baseline shortlist for Isolation Forest, OCSVM, PCA and ECOD."""
    experiment_rows: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []

    for config in IFOREST_NOVELTY_CONFIGS:
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
        recall_rows.extend(per_class_recall(eval_frame, predictions, str(config["name"])))

    for config in OCSVM_CONFIGS:
        metrics, predictions = evaluate_ocsvm(x_train, x_eval, eval_frame, config)
        experiment_rows.append(
            {
                "algorithm": "one_class_svm",
                "config_name": config["name"],
                "parameters": f"kernel=rbf, nu={config['nu']}, gamma={config['gamma']}",
                **metrics,
            }
        )
        recall_rows.extend(per_class_recall(eval_frame, predictions, str(config["name"])))

    for config in PCA_CONFIGS:
        metrics, predictions = evaluate_pca_reconstruction(x_train, x_eval, eval_frame, config)
        experiment_rows.append(
            {
                "algorithm": "pca_reconstruction",
                "config_name": config["name"],
                "parameters": (
                    f"n_components={config['n_components']}, "
                    f"threshold_percentile={config['threshold_percentile']}"
                ),
                **metrics,
            }
        )
        recall_rows.extend(per_class_recall(eval_frame, predictions, str(config["name"])))

    for config in ECOD_CONFIGS:
        metrics, predictions = evaluate_ecod(x_train, x_eval, eval_frame, config)
        experiment_rows.append(
            {
                "algorithm": "ecod",
                "config_name": config["name"],
                "parameters": f"threshold_percentile={config['threshold_percentile']}",
                **metrics,
            }
        )
        recall_rows.extend(per_class_recall(eval_frame, predictions, str(config["name"])))

    return pd.DataFrame(experiment_rows), pd.DataFrame(recall_rows)


def run_trends(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the parameter sweeps used to show novelty-baseline trends."""
    iforest_rows: list[dict[str, object]] = []
    ocsvm_nu_rows: list[dict[str, object]] = []
    ocsvm_gamma_rows: list[dict[str, object]] = []
    pca_component_rows: list[dict[str, object]] = []
    pca_threshold_rows: list[dict[str, object]] = []
    ecod_rows: list[dict[str, object]] = []

    for config in IFOREST_THRESHOLD_TREND:
        metrics, _ = evaluate_iforest(x_train, x_eval, eval_frame, config)
        iforest_rows.append(
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

    for config in OCSVM_NU_TREND:
        metrics, _ = evaluate_ocsvm(x_train, x_eval, eval_frame, config)
        ocsvm_nu_rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "nu": config["nu"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )

    for config in OCSVM_GAMMA_TREND:
        metrics, _ = evaluate_ocsvm(x_train, x_eval, eval_frame, config)
        ocsvm_gamma_rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "gamma": config["gamma"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )

    for config in PCA_COMPONENT_TREND:
        metrics, _ = evaluate_pca_reconstruction(x_train, x_eval, eval_frame, config)
        pca_component_rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "n_components": config["n_components"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )

    for config in PCA_THRESHOLD_TREND:
        metrics, _ = evaluate_pca_reconstruction(x_train, x_eval, eval_frame, config)
        pca_threshold_rows.append(
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

    for config in ECOD_THRESHOLD_TREND:
        metrics, _ = evaluate_ecod(x_train, x_eval, eval_frame, config)
        ecod_rows.append(
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

    return (
        pd.DataFrame(iforest_rows),
        pd.DataFrame(ocsvm_nu_rows),
        pd.DataFrame(ocsvm_gamma_rows),
        pd.DataFrame(pca_component_rows),
        pd.DataFrame(pca_threshold_rows),
        pd.DataFrame(ecod_rows),
    )


def run_ocsvm_refined_grid(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Run a local OCSVM sweep around the most promising parameter region."""
    rows: list[dict[str, object]] = []

    for config in OCSVM_REFINED_GRID:
        metrics, _ = evaluate_ocsvm(x_train, x_eval, eval_frame, config)
        rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "nu": config["nu"],
                "gamma": config["gamma"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "runtime_seconds": metrics["runtime_seconds"],
            }
        )

    return pd.DataFrame(rows)


def run_pca_refined_grid(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    eval_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Run a local PCA sweep around the most promising component/threshold region."""

    rows: list[dict[str, object]] = []

    for config in PCA_REFINED_GRID:
        metrics, _ = evaluate_pca_reconstruction(x_train, x_eval, eval_frame, config)
        rows.append(
            {
                "algorithm": config["algorithm"],
                "setting": config["setting"],
                "n_components": config["n_components"],
                "threshold_percentile": config["threshold_percentile"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "runtime_seconds": metrics["runtime_seconds"],
            }
        )

    return pd.DataFrame(rows)
