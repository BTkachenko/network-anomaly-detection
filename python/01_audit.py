from __future__ import annotations

from data_utils import RESULTS_DIR, dataset_audit


def main() -> None:
    """Create audit tables for the IoT-DIAD flow-based dataset."""
    audit_files, summary = dataset_audit()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    audit_files.to_csv(RESULTS_DIR / "python_audit_files.csv", index=False)
    summary_frame = (
        audit_files.groupby("coarse_label", as_index=False)["rows"].sum().sort_values("rows", ascending=False)
    )
    summary_frame.to_csv(RESULTS_DIR / "python_audit_by_label.csv", index=False)

    with open(RESULTS_DIR / "python_audit_summary.csv", "w", encoding="utf-8", newline="") as handle:
        handle.write("item,value\n")
        for key, value in summary.items():
            handle.write(f"{key},{value}\n")

    print("Zapisano:")
    print(RESULTS_DIR / "python_audit_files.csv")
    print(RESULTS_DIR / "python_audit_by_label.csv")
    print(RESULTS_DIR / "python_audit_summary.csv")


if __name__ == "__main__":
    main()
