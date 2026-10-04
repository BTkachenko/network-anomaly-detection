from __future__ import annotations

import argparse

from data_utils import RESULTS_DIR, load_samples, prepare_features
from novelty_stage import run_final_configs, run_ocsvm_refined_grid, run_pca_refined_grid, run_trends


def main() -> None:
    """Run the first novelty-detection baseline stage on the shared sample."""
    parser = argparse.ArgumentParser(
        description="Run Isolation Forest and One-Class SVM on the shared novelty-detection sample."
    )
    args = parser.parse_args()
    _ = args

    train, eval_frame = load_samples()
    x_train, x_eval, used_columns = prepare_features(train, eval_frame)

    experiments, recall_by_class = run_final_configs(x_train, x_eval, eval_frame)
    (
        iforest_trend,
        ocsvm_nu_trend,
        ocsvm_gamma_trend,
        pca_component_trend,
        pca_threshold_trend,
        ecod_threshold_trend,
    ) = run_trends(x_train, x_eval, eval_frame)
    ocsvm_refined_grid = run_ocsvm_refined_grid(x_train, x_eval, eval_frame)
    pca_refined_grid = run_pca_refined_grid(x_train, x_eval, eval_frame)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments.insert(0, "feature_count", len(used_columns))
    experiments.to_csv(RESULTS_DIR / "python_novelty_experiments.csv", index=False)
    recall_by_class.to_csv(RESULTS_DIR / "python_novelty_recall_by_class.csv", index=False)
    iforest_trend.to_csv(RESULTS_DIR / "python_novelty_iforest_threshold_trend.csv", index=False)
    ocsvm_nu_trend.to_csv(RESULTS_DIR / "python_novelty_ocsvm_nu_trend.csv", index=False)
    ocsvm_gamma_trend.to_csv(RESULTS_DIR / "python_novelty_ocsvm_gamma_trend.csv", index=False)
    ocsvm_refined_grid.to_csv(RESULTS_DIR / "python_novelty_ocsvm_refined_grid.csv", index=False)
    pca_component_trend.to_csv(RESULTS_DIR / "python_novelty_pca_component_trend.csv", index=False)
    pca_threshold_trend.to_csv(RESULTS_DIR / "python_novelty_pca_threshold_trend.csv", index=False)
    pca_refined_grid.to_csv(RESULTS_DIR / "python_novelty_pca_refined_grid.csv", index=False)
    ecod_threshold_trend.to_csv(RESULTS_DIR / "python_novelty_ecod_threshold_trend.csv", index=False)

    print(experiments.sort_values(["algorithm", "f1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
