from __future__ import annotations

import pandas as pd

from clustering_stage import (
    build_birch_configs,
    build_density_configs,
    build_gmm_configs,
    estimate_density_reference,
    per_class_recall,
    run_algorithm_config,
)
from data_utils import RESULTS_DIR, load_samples, prepare_features


def main() -> None:
    """Run additional clustering-oriented baselines on the shared anomaly-detection sample."""
    train, eval_frame = load_samples()
    x_train, x_eval, used_columns = prepare_features(train, eval_frame)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    density_reference = estimate_density_reference(x_train, neighbor_count=10)
    density_reference.to_csv(RESULTS_DIR / "python_clustering_density_reference.csv", index=False)

    configs = (
        [("birch", config) for config in build_birch_configs()]
        + [("gmm", config) for config in build_gmm_configs()]
        + [("dbscan", config) for config in build_density_configs(density_reference, "dbscan")]
        + [("optics", config) for config in build_density_configs(density_reference, "optics")]
    )

    experiment_rows: list[dict[str, str | float | int]] = []
    recall_rows: list[dict[str, str | float]] = []

    for algorithm, config in configs:
        metrics, predictions = run_algorithm_config(algorithm, x_train, x_eval, eval_frame, config)
        metrics["feature_count"] = len(used_columns)
        experiment_rows.append(metrics)
        recall_rows.extend(per_class_recall(eval_frame, predictions, str(config["name"])))

    experiments = pd.DataFrame(experiment_rows)
    recall_by_class = pd.DataFrame(recall_rows)

    experiments.to_csv(RESULTS_DIR / "python_clustering_extensions_experiments.csv", index=False)
    recall_by_class.to_csv(RESULTS_DIR / "python_clustering_extensions_recall_by_class.csv", index=False)

    print("Dodatkowe metody klasteryzacyjne i gestosciowe:")
    print(experiments.sort_values(["algorithm", "f1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
