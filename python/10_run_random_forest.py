from __future__ import annotations

from data_utils import RESULTS_DIR, load_supervised_samples
from supervised_stage import (
    run_random_forest_binary_benchmark,
    run_random_forest_multiclass_benchmark,
)


def main() -> None:
    """Run the first supervised benchmark based on Random Forest."""
    train_frame, validation_frame, test_frame = load_supervised_samples()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    binary_results = run_random_forest_binary_benchmark(train_frame, validation_frame, test_frame)
    multiclass_results, recall_by_class, confusion = run_random_forest_multiclass_benchmark(
        train_frame, validation_frame, test_frame
    )

    binary_results.to_csv(RESULTS_DIR / "python_rf_binary_results.csv", index=False)
    multiclass_results.to_csv(RESULTS_DIR / "python_rf_multiclass_results.csv", index=False)
    recall_by_class.to_csv(RESULTS_DIR / "python_rf_multiclass_recall_by_class.csv", index=False)
    confusion.to_csv(RESULTS_DIR / "python_rf_multiclass_confusion_matrix.csv", index=False)

    print("Random Forest - wynik binarny:")
    print(binary_results.sort_values(["split", "f1"], ascending=[True, False]).to_string(index=False))
    print()
    print("Random Forest - wynik 7 klas:")
    print(
        multiclass_results.sort_values(["split", "macro_f1"], ascending=[True, False]).to_string(index=False)
    )


if __name__ == "__main__":
    main()
