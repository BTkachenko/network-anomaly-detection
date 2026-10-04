from __future__ import annotations

import argparse

import pandas as pd

from data_utils import RESULTS_DIR, load_samples, prepare_features
from hdbscan_stage import (
    HDBSCAN_PCA_COMPONENTS,
    build_hdbscan_space,
    run_final_configs,
    run_trends,
    summarize_by_class,
    summarize_by_cluster,
)


def main() -> None:
    """Run HDBSCAN as a structure-analysis step on the mixed evaluation sample."""

    parser = argparse.ArgumentParser(
        description="Run HDBSCAN on the mixed sample and save structural summaries."
    )
    args = parser.parse_args()
    _ = args

    train, eval_frame = load_samples()
    _, x_eval, used_columns = prepare_features(train, eval_frame)
    x_projected, explained_variance = build_hdbscan_space(x_eval)

    final_summary, assignments = run_final_configs(x_projected, eval_frame)
    cluster_size_trend, min_samples_trend = run_trends(x_projected, eval_frame)
    class_summary = summarize_by_class(assignments)
    cluster_summary = summarize_by_cluster(assignments)

    representation_summary = pd.DataFrame(
        [
            {"item": "input_feature_count", "value": len(used_columns)},
            {"item": "pca_components", "value": HDBSCAN_PCA_COMPONENTS},
            {"item": "pca_cumulative_explained_variance", "value": explained_variance},
            {"item": "mixed_rows", "value": len(eval_frame)},
        ]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_summary.to_csv(RESULTS_DIR / "python_hdbscan_experiments.csv", index=False)
    assignments.to_csv(RESULTS_DIR / "python_hdbscan_assignments.csv", index=False)
    class_summary.to_csv(RESULTS_DIR / "python_hdbscan_class_summary.csv", index=False)
    cluster_summary.to_csv(RESULTS_DIR / "python_hdbscan_cluster_summary.csv", index=False)
    cluster_size_trend.to_csv(RESULTS_DIR / "python_hdbscan_cluster_size_trend.csv", index=False)
    min_samples_trend.to_csv(RESULTS_DIR / "python_hdbscan_min_samples_trend.csv", index=False)
    representation_summary.to_csv(RESULTS_DIR / "python_hdbscan_representation_summary.csv", index=False)

    print(
        final_summary.sort_values(["noise_ratio", "clusters_found"], ascending=[True, False]).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
