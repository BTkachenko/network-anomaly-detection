from __future__ import annotations

import argparse

from data_utils import (
    EVAL_SAMPLE_PATH,
    RESULTS_DIR,
    TRAIN_SAMPLE_PATH,
    ensure_sample_files,
    load_samples,
    sample_summary,
)


def main() -> None:
    """Create the shared sample used in the experiments."""
    parser = argparse.ArgumentParser(description="Build the shared 30k / 12k / 14k sample.")
    parser.add_argument(
        "--force", action="store_true", help="Rebuild sample files even if they already exist."
    )
    args = parser.parse_args()

    train_path, eval_path = ensure_sample_files(force=args.force)
    train, eval_frame = load_samples()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "python_sample_summary.csv"
    sample_summary(train, eval_frame).to_csv(summary_path, index=False)

    print("Zapisano probki:")
    print(train_path if train_path else TRAIN_SAMPLE_PATH)
    print(eval_path if eval_path else EVAL_SAMPLE_PATH)
    print(summary_path)


if __name__ == "__main__":
    main()
