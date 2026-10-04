from __future__ import annotations

import argparse

import pandas as pd

from data_utils import RESULTS_DIR, load_supervised_samples_with_sizes, supervised_sample_summary
from supervised_stage import (
    run_hist_gradient_boosting_binary_benchmark,
    run_random_forest_binary_benchmark,
)


DEFAULT_BENIGN_SIZES = [30_000, 50_000, 100_000]


def parse_sizes(values: list[str]) -> list[int]:
    """Parse the requested benign train sizes."""
    sizes = sorted({int(value) for value in values})
    if not sizes:
        raise ValueError("Lista rozmiarow benign nie moze byc pusta.")
    return sizes


def build_scale_summary(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize the best points by F1 and by recall under FPR <= 0.10."""
    test_only = raw_results.loc[raw_results["split"] == "test"].copy()
    rows: list[dict[str, str | int | float]] = []

    for (model_family, train_benign_rows), group in test_only.groupby(
        ["model_family", "train_benign_rows"], sort=True
    ):
        best_f1 = group.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
        rows.append(
            {
                "model_family": model_family,
                "train_benign_rows": int(train_benign_rows),
                "selection_rule": "best_f1",
                "config_name": str(best_f1["config_name"]),
                "precision": float(best_f1["precision"]),
                "recall": float(best_f1["recall"]),
                "f1": float(best_f1["f1"]),
                "fpr": float(best_f1["fpr"]),
                "runtime_seconds": float(best_f1["runtime_seconds"]),
            }
        )

        practical = group.loc[group["fpr"] <= 0.10].copy()
        if practical.empty:
            continue

        best_practical = practical.sort_values(
            ["recall", "f1", "precision"], ascending=[False, False, False]
        ).iloc[0]
        rows.append(
            {
                "model_family": model_family,
                "train_benign_rows": int(train_benign_rows),
                "selection_rule": "best_recall_fpr_le_0.10",
                "config_name": str(best_practical["config_name"]),
                "precision": float(best_practical["precision"]),
                "recall": float(best_practical["recall"]),
                "f1": float(best_practical["f1"]),
                "fpr": float(best_practical["fpr"]),
                "runtime_seconds": float(best_practical["runtime_seconds"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Compare supervised binary results for larger benign train sizes."""
    parser = argparse.ArgumentParser(
        description="Compare supervised binary benchmarks for different benign train sizes."
    )
    parser.add_argument(
        "--benign-sizes",
        nargs="+",
        default=[str(size) for size in DEFAULT_BENIGN_SIZES],
        help="Lista rozmiarow benign train do porownania, np. 30000 50000 100000.",
    )
    args = parser.parse_args()

    benign_sizes = parse_sizes(args.benign_sizes)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []

    for benign_rows in benign_sizes:
        train_frame, validation_frame, test_frame = load_supervised_samples_with_sizes(
            train_benign_rows=benign_rows
        )

        sample_summary = supervised_sample_summary(train_frame, validation_frame, test_frame).copy()
        sample_summary.insert(0, "train_benign_rows", benign_rows)
        sample_frames.append(sample_summary)

        rf_results = run_random_forest_binary_benchmark(train_frame, validation_frame, test_frame).copy()
        rf_results.insert(0, "model_family", "RandomForest")
        rf_results.insert(1, "train_benign_rows", benign_rows)
        raw_frames.append(rf_results)

        hgb_results = run_hist_gradient_boosting_binary_benchmark(
            train_frame, validation_frame, test_frame
        ).copy()
        hgb_results.insert(0, "model_family", "HistGradientBoosting")
        hgb_results.insert(1, "train_benign_rows", benign_rows)
        raw_frames.append(hgb_results)

    raw_results = pd.concat(raw_frames, ignore_index=True)
    sample_results = pd.concat(sample_frames, ignore_index=True)
    summary = build_scale_summary(raw_results)

    raw_path = RESULTS_DIR / "python_supervised_benign_scale_study.csv"
    summary_path = RESULTS_DIR / "python_supervised_benign_scale_summary.csv"
    sample_path = RESULTS_DIR / "python_supervised_benign_scale_sample_summary.csv"

    raw_results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    sample_results.to_csv(sample_path, index=False)

    print("Zapisano porownanie supervised dla roznych rozmiarow benign train:")
    print(raw_path)
    print(summary_path)
    print(sample_path)
    print()
    print(summary.sort_values(["model_family", "train_benign_rows", "selection_rule"]).to_string(index=False))


if __name__ == "__main__":
    main()
