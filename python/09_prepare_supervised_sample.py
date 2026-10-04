from __future__ import annotations

import argparse

from data_utils import (
    RESULTS_DIR,
    SUPERVISED_TRAIN_PATH,
    SUPERVISED_VALIDATION_PATH,
    ensure_supervised_sample_files,
    load_supervised_samples,
    supervised_sample_summary,
)


def main() -> None:
    """Build the supervised train/validation split used by later benchmarks."""
    parser = argparse.ArgumentParser(description="Build the supervised mixed train/validation split.")
    parser.add_argument(
        "--force", action="store_true", help="Rebuild supervised sample files even if they already exist."
    )
    args = parser.parse_args()

    train_path, validation_path = ensure_supervised_sample_files(force=args.force)
    train_frame, validation_frame, test_frame = load_supervised_samples()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "python_supervised_sample_summary.csv"
    supervised_sample_summary(train_frame, validation_frame, test_frame).to_csv(summary_path, index=False)

    print("Zapisano probki supervised:")
    print(train_path if train_path else SUPERVISED_TRAIN_PATH)
    print(validation_path if validation_path else SUPERVISED_VALIDATION_PATH)
    print(summary_path)


if __name__ == "__main__":
    main()
