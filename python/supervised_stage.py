from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from data_utils import EXCLUDED_COLUMNS, MODEL_RANDOM_SEED


RF_CONFIGS = [
    {
        "name": "rf_practical",
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "name": "rf_balanced",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "name": "rf_regularized",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
]


HGB_CONFIGS = [
    {
        "name": "hgb_practical",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 0.0,
    },
    {
        "name": "hgb_balanced",
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
    {
        "name": "hgb_regularized",
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 0.1,
    },
]


def prepare_supervised_features(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Prepare feature matrices for supervised models."""
    feature_columns: list[str] = []
    for column in train_frame.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if column not in {"Src Port", "Dst Port", "Protocol"} and not pd.api.types.is_numeric_dtype(
            train_frame[column]
        ):
            continue
        feature_columns.append(column)

    train_num = (
        train_frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )
    validation_num = (
        validation_frame[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    test_num = (
        test_frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )

    kept_columns = [column for column in train_num.columns if train_num[column].nunique(dropna=True) > 1]
    train_num = train_num[kept_columns]
    validation_num = validation_num[kept_columns]
    test_num = test_num[kept_columns]

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_num)
    x_validation = imputer.transform(validation_num)
    x_test = imputer.transform(test_num)
    return x_train, x_validation, x_test, kept_columns


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Return binary classification metrics for Benign vs Attack."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return macro and weighted metrics for the 7 attack classes."""
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def attack_label_order(frame: pd.DataFrame) -> list[str]:
    """Return the sorted list of attack labels used in the multiclass task."""
    return sorted(frame.loc[frame["binary_label"].astype(int) == 1, "coarse_label"].unique().tolist())


def attack_only_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only attack rows for the multiclass task."""
    return frame.loc[frame["binary_label"].astype(int) == 1].reset_index(drop=True)


def encode_attack_labels(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """Map attack class names to consecutive integer ids."""
    label_to_id = {label: index for index, label in enumerate(labels)}
    return frame["coarse_label"].map(label_to_id).astype(int).to_numpy()


def per_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    config_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Return recall for each attack class separately."""
    rows: list[dict[str, str | float]] = []
    for index, label in enumerate(labels):
        true_positive = int(((y_true == index) & (y_pred == index)).sum())
        false_negative = int(((y_true == index) & (y_pred != index)).sum())
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        rows.append(
            {
                "config_name": config_name,
                "split": split_name,
                "attack_class": label,
                "recall": recall,
            }
        )
    return pd.DataFrame(rows)


def confusion_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    config_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Return a long-form confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    rows: list[dict[str, str | int]] = []
    for true_index, true_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            rows.append(
                {
                    "config_name": config_name,
                    "split": split_name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[true_index, predicted_index]),
                }
            )
    return pd.DataFrame(rows)


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, int | str | None],
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[RandomForestClassifier, float]:
    """Fit one Random Forest configuration and return model with fit time."""
    started = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=None if config["max_depth"] is None else int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=str(config["max_features"]),
        class_weight=None if config["class_weight"] is None else str(config["class_weight"]),
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model, time.perf_counter() - started


def balanced_sample_weights(y_train: np.ndarray) -> np.ndarray:
    """Return balanced sample weights for supervised training."""
    return compute_sample_weight(class_weight="balanced", y=y_train)


def train_hist_gradient_boosting(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, int | float | str],
    random_seed: int = MODEL_RANDOM_SEED,
) -> tuple[HistGradientBoostingClassifier, float]:
    """Fit one HistGradientBoosting configuration and return model with fit time."""
    started = time.perf_counter()
    model = HistGradientBoostingClassifier(
        learning_rate=float(config["learning_rate"]),
        max_iter=int(config["max_iter"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        early_stopping=False,
        random_state=random_seed,
    )
    sample_weight = balanced_sample_weights(y_train)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model, time.perf_counter() - started


def run_random_forest_binary_benchmark(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate Random Forest on the binary Benign vs Attack task."""
    x_train, x_validation, x_test, used_columns = prepare_supervised_features(
        train_frame, validation_frame, test_frame
    )
    y_train = train_frame["binary_label"].astype(int).to_numpy()
    y_validation = validation_frame["binary_label"].astype(int).to_numpy()
    y_test = test_frame["binary_label"].astype(int).to_numpy()

    rows: list[dict[str, str | float | int]] = []
    for config in RF_CONFIGS:
        model, fit_seconds = train_random_forest(x_train, y_train, config)
        for split_name, x_split, y_split in (
            ("validation", x_validation, y_validation),
            ("test", x_test, y_test),
        ):
            predictions = model.predict(x_split)
            metrics = binary_metrics(y_split, predictions)
            rows.append(
                {
                    "feature_count": len(used_columns),
                    "config_name": config["name"],
                    "split": split_name,
                    "parameters": (
                        f"n_estimators={config['n_estimators']}, max_depth={config['max_depth']}, "
                        f"min_samples_leaf={config['min_samples_leaf']}, max_features={config['max_features']}, "
                        f"class_weight={config['class_weight']}"
                    ),
                    **metrics,
                    "runtime_seconds": fit_seconds,
                }
            )
    return pd.DataFrame(rows)


def run_random_forest_multiclass_benchmark(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate Random Forest on the 7-class attack classification task."""
    attack_train = attack_only_frame(train_frame)
    attack_validation = attack_only_frame(validation_frame)
    attack_test = attack_only_frame(test_frame)

    x_train, x_validation, x_test, used_columns = prepare_supervised_features(
        attack_train, attack_validation, attack_test
    )
    labels = attack_label_order(attack_train)
    y_train = encode_attack_labels(attack_train, labels)
    y_validation = encode_attack_labels(attack_validation, labels)
    y_test = encode_attack_labels(attack_test, labels)

    result_rows: list[dict[str, str | float | int]] = []
    recall_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []

    for config in RF_CONFIGS:
        model, fit_seconds = train_random_forest(x_train, y_train, config)
        for split_name, x_split, y_split in (
            ("validation", x_validation, y_validation),
            ("test", x_test, y_test),
        ):
            predictions = model.predict(x_split)
            metrics = multiclass_metrics(y_split, predictions)
            result_rows.append(
                {
                    "feature_count": len(used_columns),
                    "config_name": config["name"],
                    "split": split_name,
                    "parameters": (
                        f"n_estimators={config['n_estimators']}, max_depth={config['max_depth']}, "
                        f"min_samples_leaf={config['min_samples_leaf']}, max_features={config['max_features']}, "
                        f"class_weight={config['class_weight']}"
                    ),
                    **metrics,
                    "runtime_seconds": fit_seconds,
                }
            )
            recall_frames.append(per_class_recall(y_split, predictions, labels, config["name"], split_name))
            confusion_frames.append(confusion_table(y_split, predictions, labels, config["name"], split_name))

    return (
        pd.DataFrame(result_rows),
        pd.concat(recall_frames, ignore_index=True),
        pd.concat(confusion_frames, ignore_index=True),
    )


def run_hist_gradient_boosting_binary_benchmark(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate HistGradientBoosting on the binary Benign vs Attack task."""
    x_train, x_validation, x_test, used_columns = prepare_supervised_features(
        train_frame, validation_frame, test_frame
    )
    y_train = train_frame["binary_label"].astype(int).to_numpy()
    y_validation = validation_frame["binary_label"].astype(int).to_numpy()
    y_test = test_frame["binary_label"].astype(int).to_numpy()

    rows: list[dict[str, str | float | int]] = []
    for config in HGB_CONFIGS:
        model, fit_seconds = train_hist_gradient_boosting(x_train, y_train, config)
        for split_name, x_split, y_split in (
            ("validation", x_validation, y_validation),
            ("test", x_test, y_test),
        ):
            predictions = model.predict(x_split)
            metrics = binary_metrics(y_split, predictions)
            rows.append(
                {
                    "feature_count": len(used_columns),
                    "config_name": config["name"],
                    "split": split_name,
                    "parameters": (
                        f"learning_rate={config['learning_rate']}, max_iter={config['max_iter']}, "
                        f"max_leaf_nodes={config['max_leaf_nodes']}, min_samples_leaf={config['min_samples_leaf']}, "
                        f"l2_regularization={config['l2_regularization']}"
                    ),
                    **metrics,
                    "runtime_seconds": fit_seconds,
                }
            )
    return pd.DataFrame(rows)


def run_hist_gradient_boosting_multiclass_benchmark(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate HistGradientBoosting on the 7-class attack classification task."""
    attack_train = attack_only_frame(train_frame)
    attack_validation = attack_only_frame(validation_frame)
    attack_test = attack_only_frame(test_frame)

    x_train, x_validation, x_test, used_columns = prepare_supervised_features(
        attack_train, attack_validation, attack_test
    )
    labels = attack_label_order(attack_train)
    y_train = encode_attack_labels(attack_train, labels)
    y_validation = encode_attack_labels(attack_validation, labels)
    y_test = encode_attack_labels(attack_test, labels)

    result_rows: list[dict[str, str | float | int]] = []
    recall_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []

    for config in HGB_CONFIGS:
        model, fit_seconds = train_hist_gradient_boosting(x_train, y_train, config)
        for split_name, x_split, y_split in (
            ("validation", x_validation, y_validation),
            ("test", x_test, y_test),
        ):
            predictions = model.predict(x_split)
            metrics = multiclass_metrics(y_split, predictions)
            result_rows.append(
                {
                    "feature_count": len(used_columns),
                    "config_name": config["name"],
                    "split": split_name,
                    "parameters": (
                        f"learning_rate={config['learning_rate']}, max_iter={config['max_iter']}, "
                        f"max_leaf_nodes={config['max_leaf_nodes']}, min_samples_leaf={config['min_samples_leaf']}, "
                        f"l2_regularization={config['l2_regularization']}"
                    ),
                    **metrics,
                    "runtime_seconds": fit_seconds,
                }
            )
            recall_frames.append(per_class_recall(y_split, predictions, labels, config["name"], split_name))
            confusion_frames.append(confusion_table(y_split, predictions, labels, config["name"], split_name))

    return (
        pd.DataFrame(result_rows),
        pd.concat(recall_frames, ignore_index=True),
        pd.concat(confusion_frames, ignore_index=True),
    )
