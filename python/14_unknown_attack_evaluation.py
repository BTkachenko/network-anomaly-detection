from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from data_utils import EXCLUDED_COLUMNS, RESULTS_DIR, load_supervised_samples
from supervised_stage import (
    HGB_CONFIGS,
    RF_CONFIGS,
    binary_metrics,
    train_hist_gradient_boosting,
    train_random_forest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLUSTERING_REFERENCE_MODELS = [
    {
        "model": "kmeans_mid_fpr",
        "group": "clustering",
        "metrics_path": RESULTS_DIR / "python_experiments.csv",
        "recall_path": RESULTS_DIR / "python_recall_by_class.csv",
    },
    {
        "model": "birch_refined_t0.75_p94",
        "group": "clustering",
        "metrics_path": RESULTS_DIR / "python_birch_refined_grid.csv",
        "recall_path": RESULTS_DIR / "python_clustering_refined_recall_by_class.csv",
    },
    {
        "model": "dbscan_refined_eps0.38_m8",
        "group": "clustering",
        "metrics_path": RESULTS_DIR / "python_dbscan_refined_grid.csv",
        "recall_path": RESULTS_DIR / "python_clustering_refined_recall_by_class.csv",
    },
]


def prepare_train_test_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare feature matrices for binary unknown-attack evaluation."""
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
    test_num = (
        test_frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )

    kept_columns = [column for column in train_num.columns if train_num[column].nunique(dropna=True) > 1]
    train_num = train_num[kept_columns]
    test_num = test_num[kept_columns]

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_num)
    x_test = imputer.transform(test_num)
    return x_train, x_test, kept_columns


def attack_binary_recall(eval_frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Return binary attack recall for every attack class on the mixed test set."""
    rows: list[dict[str, str | float]] = []
    attack_only = eval_frame.loc[eval_frame["binary_label"].astype(int) == 1].copy()
    attack_only["prediction"] = predictions[eval_frame["binary_label"].astype(int).to_numpy() == 1]

    for label, frame in attack_only.groupby("coarse_label", sort=True):
        truth = frame["binary_label"].astype(int).to_numpy()
        pred = frame["prediction"].astype(int).to_numpy()
        tp = int(((truth == 1) & (pred == 1)).sum())
        fn = int(((truth == 1) & (pred == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({"attack_class": label, "recall": recall})
    return pd.DataFrame(rows)


def summarize_unknown(
    model_name: str,
    model_group: str,
    held_out_class: str,
    metrics: dict[str, float | int],
    recall_frame: pd.DataFrame,
    feature_count: int,
    train_rows: int | None = None,
) -> dict[str, str | float | int]:
    """Build one summary row for a leave-one-class-out experiment."""
    unknown_row = recall_frame.loc[recall_frame["attack_class"] == held_out_class].iloc[0]
    known_rows = recall_frame.loc[recall_frame["attack_class"] != held_out_class]
    known_recall_mean = float(known_rows["recall"].mean())
    unknown_recall = float(unknown_row["recall"])

    return {
        "model": model_name,
        "model_group": model_group,
        "held_out_class": held_out_class,
        "feature_count": feature_count,
        "train_rows": int(train_rows) if train_rows is not None else "",
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "fpr": float(metrics["fpr"]),
        "known_recall_mean": known_recall_mean,
        "unknown_recall": unknown_recall,
        "unknown_minus_known": unknown_recall - known_recall_mean,
    }


def clustering_unknown_summary(test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse static clustering recalls because these models always train on Benign only."""
    rows: list[dict[str, str | float | int]] = []
    recall_rows: list[pd.DataFrame] = []
    held_out_classes = sorted(
        test_frame.loc[test_frame["binary_label"].astype(int) == 1, "coarse_label"].unique().tolist()
    )

    for model_info in CLUSTERING_REFERENCE_MODELS:
        metrics_frame = pd.read_csv(model_info["metrics_path"])
        recall_frame = pd.read_csv(model_info["recall_path"])

        metrics_row = metrics_frame.loc[metrics_frame["config_name"] == model_info["model"]].iloc[0]
        model_recall = recall_frame.loc[
            recall_frame["config_name"] == model_info["model"], ["attack_class", "recall"]
        ].copy()
        model_recall["model"] = model_info["model"]
        model_recall["held_out_class"] = ""
        recall_rows.append(model_recall.copy())

        metrics = {
            "precision": float(metrics_row["precision"]),
            "recall": float(metrics_row["recall"]),
            "f1": float(metrics_row["f1"]),
            "fpr": float(metrics_row["fpr"]),
        }

        for held_out_class in held_out_classes:
            rows.append(
                summarize_unknown(
                    model_name=model_info["model"],
                    model_group=model_info["group"],
                    held_out_class=held_out_class,
                    metrics=metrics,
                    recall_frame=model_recall.loc[:, ["attack_class", "recall"]],
                    feature_count=int(metrics_row["feature_count"]),
                )
            )

    return pd.DataFrame(rows), pd.concat(recall_rows, ignore_index=True)


def train_supervised_unknown_models(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave one attack class out of supervised training and measure binary detection on test."""
    combined_train = pd.concat([train_frame, validation_frame], ignore_index=True)
    y_test = test_frame["binary_label"].astype(int).to_numpy()
    held_out_classes = sorted(
        test_frame.loc[test_frame["binary_label"].astype(int) == 1, "coarse_label"].unique().tolist()
    )

    rf_config = next(config for config in RF_CONFIGS if config["name"] == "rf_practical")
    hgb_config = next(config for config in HGB_CONFIGS if config["name"] == "hgb_balanced")

    rows: list[dict[str, str | float | int]] = []
    recall_rows: list[pd.DataFrame] = []

    for held_out_class in held_out_classes:
        known_train = combined_train.loc[
            ~(
                (combined_train["binary_label"].astype(int) == 1)
                & (combined_train["coarse_label"] == held_out_class)
            )
        ].reset_index(drop=True)

        x_train, x_test, used_columns = prepare_train_test_features(known_train, test_frame)
        y_train = known_train["binary_label"].astype(int).to_numpy()

        rf_model, rf_seconds = train_random_forest(x_train, y_train, rf_config)
        rf_predictions = rf_model.predict(x_test)
        rf_metrics = binary_metrics(y_test, rf_predictions)
        rf_recall = attack_binary_recall(test_frame, rf_predictions)
        rf_recall["model"] = "rf_practical"
        rf_recall["held_out_class"] = held_out_class
        recall_rows.append(rf_recall)
        rows.append(
            {
                **summarize_unknown(
                    model_name="rf_practical",
                    model_group="supervised",
                    held_out_class=held_out_class,
                    metrics=rf_metrics,
                    recall_frame=rf_recall.loc[:, ["attack_class", "recall"]],
                    feature_count=len(used_columns),
                    train_rows=len(known_train),
                ),
                "runtime_seconds": rf_seconds,
            }
        )

        hgb_model, hgb_seconds = train_hist_gradient_boosting(x_train, y_train, hgb_config)
        hgb_predictions = hgb_model.predict(x_test)
        hgb_metrics = binary_metrics(y_test, hgb_predictions)
        hgb_recall = attack_binary_recall(test_frame, hgb_predictions)
        hgb_recall["model"] = "hgb_balanced"
        hgb_recall["held_out_class"] = held_out_class
        recall_rows.append(hgb_recall)
        rows.append(
            {
                **summarize_unknown(
                    model_name="hgb_balanced",
                    model_group="supervised",
                    held_out_class=held_out_class,
                    metrics=hgb_metrics,
                    recall_frame=hgb_recall.loc[:, ["attack_class", "recall"]],
                    feature_count=len(used_columns),
                    train_rows=len(known_train),
                ),
                "runtime_seconds": hgb_seconds,
            }
        )

    return pd.DataFrame(rows), pd.concat(recall_rows, ignore_index=True)


def aggregate_unknown(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate leave-one-class-out results per model."""
    grouped = frame.groupby(["model", "model_group"], as_index=False).agg(
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_fpr=("fpr", "mean"),
        mean_known_recall=("known_recall_mean", "mean"),
        mean_unknown_recall=("unknown_recall", "mean"),
        std_unknown_recall=("unknown_recall", "std"),
        mean_unknown_minus_known=("unknown_minus_known", "mean"),
    )
    return grouped


def main() -> None:
    """Run leave-one-attack-class-out evaluation for supervised models and compare to clustering."""
    train_frame, validation_frame, test_frame = load_supervised_samples()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    clustering_summary, clustering_recall = clustering_unknown_summary(test_frame)
    supervised_summary, supervised_recall = train_supervised_unknown_models(
        train_frame, validation_frame, test_frame
    )

    combined_summary = pd.concat([clustering_summary, supervised_summary], ignore_index=True)
    combined_recall = pd.concat([clustering_recall, supervised_recall], ignore_index=True)
    aggregate = aggregate_unknown(combined_summary)

    clustering_summary.to_csv(RESULTS_DIR / "python_unknown_attack_clustering_summary.csv", index=False)
    supervised_summary.to_csv(RESULTS_DIR / "python_unknown_attack_supervised_summary.csv", index=False)
    combined_summary.to_csv(RESULTS_DIR / "python_unknown_attack_summary.csv", index=False)
    combined_recall.to_csv(RESULTS_DIR / "python_unknown_attack_recall_by_class.csv", index=False)
    aggregate.to_csv(RESULTS_DIR / "python_unknown_attack_aggregate.csv", index=False)

    print("Unknown attack evaluation - aggregate:")
    print(
        aggregate.sort_values(["model_group", "mean_unknown_recall"], ascending=[True, False]).to_string(
            index=False
        )
    )
    print()
    print("Unknown attack evaluation - supervised details:")
    print(supervised_summary.sort_values(["model", "held_out_class"]).to_string(index=False))


if __name__ == "__main__":
    main()
