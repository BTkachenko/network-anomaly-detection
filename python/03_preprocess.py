from __future__ import annotations

from data_utils import RESULTS_DIR, load_samples, preprocessing_summary


def main() -> None:
    """Describe the preprocessing stage on the shared sample."""
    train, eval_frame = load_samples()
    summary = preprocessing_summary(train, eval_frame)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "python_preprocessing_summary.csv"
    summary.to_csv(out_path, index=False)
    print("Zapisano:")
    print(out_path)


if __name__ == "__main__":
    main()
