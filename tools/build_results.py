"""Build the README chart directly from recorded experiment CSVs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    baseline = pd.read_csv(ROOT / "results/python_experiments.csv").set_index("config_name")
    refined = pd.read_csv(ROOT / "results/python_clustering_refined_summary.csv")
    refined = refined.loc[refined["criterion"] == "max_f1"].set_index("algorithm")
    selections = [
        ("k-means", baseline.loc["kmeans_max_f1"]),
        ("BIRCH", refined.loc["birch"]),
        ("DBSCAN", refined.loc["dbscan"]),
        ("GMM", refined.loc["gmm"]),
        ("Custom Isolation Forest", baseline.loc["iforest_higher_recall"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(selections))
    ax.barh(
        positions - 0.17,
        [100 * row["recall"] for _, row in selections],
        height=0.3,
        color="#2563eb",
        label="Attack recall",
    )
    ax.barh(
        positions + 0.17,
        [100 * row["fpr"] for _, row in selections],
        height=0.3,
        color="#f59e0b",
        label="False-positive rate",
    )
    for i, (_, row) in enumerate(selections):
        for offset, key in ((-0.17, "recall"), (0.17, "fpr")):
            value = 100 * row[key]
            ax.text(value + 1, i + offset, f"{value:.1f}%", va="center", fontsize=10)
    ax.set_yticks(positions, [name for name, _ in selections])
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xlabel("Evaluation flows (%)")
    ax.set_title("Attack detection and false alarms", loc="left", fontsize=15, pad=18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="lower right", frameon=False)
    fig.text(
        0.02,
        0.01,
        "Selected configurations on the shared evaluation split; exploratory results.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output = ROOT / "docs/assets/model-comparison.svg"
    fig.savefig(output, metadata={"Date": None}, bbox_inches="tight")
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
