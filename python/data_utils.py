from __future__ import annotations

import csv
import hashlib
import heapq
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get(
        "NETWORK_ANOMALY_DATA_DIR",
        str(PROJECT_ROOT / "IoT-DIAD" / "Anomaly Detection - Flow based features"),
    )
)
SAMPLE_DIR = PROJECT_ROOT / "samples"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_SAMPLE_PATH = SAMPLE_DIR / "train_benign_30000.csv.gz"
EVAL_SAMPLE_PATH = SAMPLE_DIR / "eval_mixed_12000_14000.csv.gz"
SUPERVISED_TRAIN_PATH = SAMPLE_DIR / "supervised_train_mixed.csv.gz"
SUPERVISED_VALIDATION_PATH = SAMPLE_DIR / "supervised_validation_mixed.csv.gz"

BENIGN_ROWS = 30_000
EVAL_BENIGN_ROWS = 12_000
EVAL_ATTACK_ROWS = 14_000
TRAIN_BENIGN_SEED = 2026
EVAL_BENIGN_SEED = 2027
EVAL_ATTACK_SEED = 2028
MODEL_RANDOM_SEED = 2029
SUPERVISED_TRAIN_BENIGN_ROWS = 30_000
SUPERVISED_VALIDATION_BENIGN_ROWS = 3_000
SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS = 1_000
SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS = 300
SUPERVISED_TRAIN_BENIGN_SEED = 2030
SUPERVISED_VALIDATION_BENIGN_SEED = 2031
SUPERVISED_ATTACK_TRAIN_SEED = 2032
SUPERVISED_ATTACK_VALIDATION_SEED = 2033

HELPER_COLUMNS = ["coarse_label", "fine_label", "binary_label", "source_file", "source_row"]

EXCLUDED_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
    "coarse_label",
    "fine_label",
    "binary_label",
    "source_file",
    "source_row",
}


def canonical_header(root: Path = DATA_DIR) -> list[str]:
    """Return the first valid CSV header found in the dataset."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. Download the flow-based CIC IoT-DIAD data "
            "or set NETWORK_ANOMALY_DATA_DIR to its class-folder root."
        )
    for path in sorted(root.rglob("*.csv")):
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            row = next(csv.reader(handle), [])
        if row and row[0] == "Flow ID":
            return row
    raise FileNotFoundError("Nie znaleziono poprawnego naglowka CSV.")


def infer_labels(root: Path, path: Path) -> tuple[str, str]:
    """Infer class labels from the relative file path."""
    relative = path.relative_to(root)
    parts = relative.parts
    coarse = parts[0]
    fine = parts[1] if len(parts) > 2 else coarse
    return coarse, fine


def file_list(root: Path = DATA_DIR) -> list[tuple[Path, str, str, bool]]:
    """Return all CSV files together with inferred labels and header status."""
    header = canonical_header(root)
    files = []
    for path in sorted(root.rglob("*.csv")):
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            first_row = next(csv.reader(handle), [])
        coarse, fine = infer_labels(root, path)
        files.append((path, coarse, fine, first_row == header))
    return files


def iter_data_rows(path: Path, header_ok: bool, expected_width: int) -> tuple[int, list[str]]:
    """
    Yield only data rows from a CSV file.

    Files with a correct header skip the first line. Files from the broken
    DoS-TCP branch are treated as headerless data files and every line is
    checked against the expected column count.
    """

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        if header_ok:
            next(reader, None)

        row_index = 0
        for row in reader:
            if not row or len(row) != expected_width:
                continue
            row_index += 1
            yield row_index, row


def dataset_audit(root: Path = DATA_DIR) -> tuple[pd.DataFrame, dict]:
    """Count rows per file and summarize the dataset."""
    rows = []
    total_rows = 0
    benign_rows = 0

    header = canonical_header(root)

    for path, coarse, fine, header_ok in file_list(root):
        row_count = sum(1 for _ in iter_data_rows(path, header_ok, len(header)))

        rows.append(
            {
                "file": str(path.relative_to(root)),
                "coarse_label": coarse,
                "fine_label": fine,
                "rows": int(row_count),
                "header_ok": bool(header_ok),
            }
        )
        total_rows += row_count
        if coarse == "Benign":
            benign_rows += row_count

    frame = pd.DataFrame(rows)
    summary = {
        "csv_files": int(len(frame)),
        "rows_total": int(total_rows),
        "benign_rows": int(benign_rows),
        "attack_rows": int(total_rows - benign_rows),
        "bad_headers": int((~frame["header_ok"]).sum()) if not frame.empty else 0,
    }
    return frame, summary


def deterministic_score(
    label: str,
    source_file: str,
    source_row: int,
    row: list[str],
    seed: int = TRAIN_BENIGN_SEED,
) -> int:
    """
    Build a deterministic score from record content and provenance.

    The source file and row number are included on purpose. Without them two
    identical rows copied from different files would receive the same score and
    could not be split cleanly between train and evaluation.
    """

    message = f"{seed}\x1e{label}\x1e{source_file}\x1e{source_row}\x1e" + "\x1f".join(row)
    digest = hashlib.sha1(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_attack_targets() -> dict[str, int]:
    """Balanced evaluation target counts for attack classes."""
    per_attack = EVAL_ATTACK_ROWS // 7
    return {
        "BruteForce": per_attack,
        "DDoS": per_attack,
        "DoS": per_attack,
        "Mirai": per_attack,
        "Recon": per_attack,
        "Spoofing": per_attack,
        "Web-Based": EVAL_ATTACK_ROWS - 6 * per_attack,
    }


def build_balanced_attack_targets(rows_per_class: int) -> dict[str, int]:
    """Return equal target sizes for all seven main attack classes."""
    return {
        "BruteForce": rows_per_class,
        "DDoS": rows_per_class,
        "DoS": rows_per_class,
        "Mirai": rows_per_class,
        "Recon": rows_per_class,
        "Spoofing": rows_per_class,
        "Web-Based": rows_per_class,
    }


def collect_top_rows(
    targets: dict[str, int],
    root: Path = DATA_DIR,
    seed: int = TRAIN_BENIGN_SEED,
    exclude_keys: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Collect the lowest-scoring rows for each requested label."""
    header = canonical_header(root)
    heaps: dict[str, list[tuple[int, list[str]]]] = {label: [] for label in targets}

    excluded = exclude_keys or set()

    for path, coarse, fine, header_ok in file_list(root):
        target_size = targets.get(coarse, 0)
        if target_size <= 0:
            continue

        relative_path = str(path.relative_to(root)).replace("\\", "/")

        for row_index, row in iter_data_rows(path, header_ok, len(header)):
            key = (relative_path, row_index)
            if key in excluded:
                continue

            score = deterministic_score(
                coarse,
                relative_path,
                row_index,
                row,
                seed=seed,
            )
            enriched = row + [
                coarse,
                fine,
                "0" if coarse == "Benign" else "1",
                relative_path,
                str(row_index),
            ]
            item = (-score, enriched)
            heap = heaps[coarse]

            if len(heap) < target_size:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)

    columns = header + HELPER_COLUMNS
    frames = []
    for label in targets:
        selected = sorted(heaps[label], key=lambda item: -item[0])
        if not selected:
            continue
        frames.append(pd.DataFrame([item[1] for item in selected], columns=columns))

    if not frames:
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True)


def build_sample_frames(root: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a disjoint train/eval split for the 30k / 12k / 14k setup."""
    train = collect_top_rows({"Benign": BENIGN_ROWS}, root=root, seed=TRAIN_BENIGN_SEED)
    train = train.sort_values(["source_file", "source_row"]).reset_index(drop=True)

    train_keys = {
        (str(source_file), int(source_row))
        for source_file, source_row in train.loc[:, ["source_file", "source_row"]].itertuples(
            index=False, name=None
        )
    }
    eval_benign = collect_top_rows(
        {"Benign": EVAL_BENIGN_ROWS},
        root=root,
        seed=EVAL_BENIGN_SEED,
        exclude_keys=train_keys,
    )
    eval_benign = eval_benign.sort_values(["source_file", "source_row"]).reset_index(drop=True)

    attack_frame = collect_top_rows(build_attack_targets(), root=root, seed=EVAL_ATTACK_SEED)
    attack_frame = attack_frame.sort_values(["coarse_label", "source_file", "source_row"]).reset_index(
        drop=True
    )

    eval_frame = pd.concat([eval_benign, attack_frame], ignore_index=True)
    return train.reset_index(drop=True), eval_frame.reset_index(drop=True)


def frame_keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    """Return the provenance keys used to keep samples disjoint."""
    return {
        (str(source_file), int(source_row))
        for source_file, source_row in frame.loc[:, ["source_file", "source_row"]].itertuples(
            index=False, name=None
        )
    }


def build_supervised_frames(root: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation splits for supervised learning.

    The test split stays equal to the shared evaluation sample used earlier,
    so supervised and unsupervised models can still be compared on the same
    mixed holdout.
    """
    return build_supervised_frames_with_sizes(
        train_benign_rows=SUPERVISED_TRAIN_BENIGN_ROWS,
        validation_benign_rows=SUPERVISED_VALIDATION_BENIGN_ROWS,
        attack_train_rows_per_class=SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS,
        attack_validation_rows_per_class=SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS,
        root=root,
    )


def build_supervised_frames_with_sizes(
    train_benign_rows: int,
    validation_benign_rows: int = SUPERVISED_VALIDATION_BENIGN_ROWS,
    attack_train_rows_per_class: int = SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS,
    attack_validation_rows_per_class: int = SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS,
    root: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation splits for supervised learning with configurable
    benign train size.

    The attack part stays balanced by class. The shared mixed test split is kept
    unchanged so that only one factor changes between experiments.
    """

    _, test_frame = load_samples()
    test_frame = test_frame.sort_values(
        ["binary_label", "coarse_label", "source_file", "source_row"]
    ).reset_index(drop=True)
    test_keys = frame_keys(test_frame)

    train_targets = {"Benign": train_benign_rows}
    train_targets.update(build_balanced_attack_targets(attack_train_rows_per_class))
    train_frame = collect_top_rows(
        train_targets,
        root=root,
        seed=SUPERVISED_TRAIN_BENIGN_SEED,
        exclude_keys=test_keys,
    )
    train_frame = train_frame.sort_values(
        ["binary_label", "coarse_label", "source_file", "source_row"]
    ).reset_index(drop=True)
    train_keys = frame_keys(train_frame)

    validation_targets = {"Benign": validation_benign_rows}
    validation_targets.update(build_balanced_attack_targets(attack_validation_rows_per_class))
    validation_frame = collect_top_rows(
        validation_targets,
        root=root,
        seed=SUPERVISED_VALIDATION_BENIGN_SEED,
        exclude_keys=test_keys | train_keys,
    )
    validation_frame = validation_frame.sort_values(
        ["binary_label", "coarse_label", "source_file", "source_row"]
    ).reset_index(drop=True)

    return train_frame, validation_frame, test_frame


def ensure_sample_files(force: bool = False) -> tuple[Path, Path]:
    """Create the shared sample files if they do not exist yet."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if force or not TRAIN_SAMPLE_PATH.exists() or not EVAL_SAMPLE_PATH.exists():
        train, eval_frame = build_sample_frames()
        train.to_csv(TRAIN_SAMPLE_PATH, index=False, compression="gzip")
        eval_frame.to_csv(EVAL_SAMPLE_PATH, index=False, compression="gzip")

    return TRAIN_SAMPLE_PATH, EVAL_SAMPLE_PATH


def train_sample_path_with_size(train_benign_rows: int) -> Path:
    """Return the cached benign-train sample path for a requested size."""
    if train_benign_rows == BENIGN_ROWS:
        return TRAIN_SAMPLE_PATH
    return SAMPLE_DIR / f"train_benign_{train_benign_rows}_fixed_eval.csv.gz"


def build_train_frame_with_size(
    train_benign_rows: int,
    root: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Build a benign-only training frame with a fixed shared evaluation split.

    The 30k sample already exists. Larger samples extend it with additional
    benign rows selected from the remaining pool after removing the current
    evaluation benign rows, so comparisons stay on the same test set.
    """
    if train_benign_rows < BENIGN_ROWS:
        raise ValueError(f"Obecny helper obsluguje tylko rozmiary >= {BENIGN_ROWS}.")

    base_train, eval_frame = load_samples()
    if train_benign_rows == len(base_train):
        return base_train

    eval_benign = eval_frame.loc[eval_frame["binary_label"].astype(int) == 0].copy()
    excluded = frame_keys(base_train) | frame_keys(eval_benign)
    additional_rows = train_benign_rows - len(base_train)
    extra_frame = collect_top_rows(
        {"Benign": additional_rows}, root=root, seed=TRAIN_BENIGN_SEED, exclude_keys=excluded
    )

    combined = pd.concat([base_train, extra_frame], ignore_index=True)
    combined = combined.sort_values(["source_file", "source_row"]).reset_index(drop=True)
    return combined


def ensure_train_sample_file_with_size(
    train_benign_rows: int,
    force: bool = False,
) -> Path:
    """Create a benign-only training sample for the requested size if needed."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_sample_files(force=False)

    train_path = train_sample_path_with_size(train_benign_rows)
    if force or not train_path.exists():
        train_frame = build_train_frame_with_size(train_benign_rows)
        train_frame.to_csv(train_path, index=False, compression="gzip")
    return train_path


def load_clustering_samples_with_train_size(
    train_benign_rows: int,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a benign-only training sample of the requested size and the shared eval split."""
    train_path = ensure_train_sample_file_with_size(train_benign_rows, force=force)
    train = pd.read_csv(train_path, low_memory=False)
    eval_frame = pd.read_csv(EVAL_SAMPLE_PATH, low_memory=False)
    return train, eval_frame


def ensure_supervised_sample_files(force: bool = False) -> tuple[Path, Path]:
    """Create the supervised train and validation sample files if needed."""
    return ensure_supervised_sample_files_with_sizes(
        train_benign_rows=SUPERVISED_TRAIN_BENIGN_ROWS,
        validation_benign_rows=SUPERVISED_VALIDATION_BENIGN_ROWS,
        attack_train_rows_per_class=SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS,
        attack_validation_rows_per_class=SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS,
        force=force,
    )


def supervised_sample_paths(
    train_benign_rows: int,
    validation_benign_rows: int = SUPERVISED_VALIDATION_BENIGN_ROWS,
    attack_train_rows_per_class: int = SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS,
    attack_validation_rows_per_class: int = SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS,
) -> tuple[Path, Path]:
    """Return file paths for a supervised split with the given sizes."""
    if (
        train_benign_rows == SUPERVISED_TRAIN_BENIGN_ROWS
        and validation_benign_rows == SUPERVISED_VALIDATION_BENIGN_ROWS
        and attack_train_rows_per_class == SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS
        and attack_validation_rows_per_class == SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS
    ):
        return SUPERVISED_TRAIN_PATH, SUPERVISED_VALIDATION_PATH

    suffix = (
        f"_b{train_benign_rows}"
        f"_vb{validation_benign_rows}"
        f"_a{attack_train_rows_per_class}"
        f"_va{attack_validation_rows_per_class}"
    )
    train_path = SAMPLE_DIR / f"supervised_train_mixed{suffix}.csv.gz"
    validation_path = SAMPLE_DIR / f"supervised_validation_mixed{suffix}.csv.gz"
    return train_path, validation_path


def ensure_supervised_sample_files_with_sizes(
    train_benign_rows: int,
    validation_benign_rows: int = SUPERVISED_VALIDATION_BENIGN_ROWS,
    attack_train_rows_per_class: int = SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS,
    attack_validation_rows_per_class: int = SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS,
    force: bool = False,
) -> tuple[Path, Path]:
    """Create supervised sample files for an arbitrary benign train size."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_sample_files(force=False)

    train_path, validation_path = supervised_sample_paths(
        train_benign_rows=train_benign_rows,
        validation_benign_rows=validation_benign_rows,
        attack_train_rows_per_class=attack_train_rows_per_class,
        attack_validation_rows_per_class=attack_validation_rows_per_class,
    )

    if force or not train_path.exists() or not validation_path.exists():
        train_frame, validation_frame, _ = build_supervised_frames_with_sizes(
            train_benign_rows=train_benign_rows,
            validation_benign_rows=validation_benign_rows,
            attack_train_rows_per_class=attack_train_rows_per_class,
            attack_validation_rows_per_class=attack_validation_rows_per_class,
        )
        train_frame.to_csv(train_path, index=False, compression="gzip")
        validation_frame.to_csv(validation_path, index=False, compression="gzip")

    return train_path, validation_path


def load_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared train and evaluation sample files."""
    ensure_sample_files()
    train = pd.read_csv(TRAIN_SAMPLE_PATH, low_memory=False)
    eval_frame = pd.read_csv(EVAL_SAMPLE_PATH, low_memory=False)
    return train, eval_frame


def load_supervised_samples() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the supervised train/validation sample and the shared mixed test."""
    return load_supervised_samples_with_sizes(
        train_benign_rows=SUPERVISED_TRAIN_BENIGN_ROWS,
        validation_benign_rows=SUPERVISED_VALIDATION_BENIGN_ROWS,
        attack_train_rows_per_class=SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS,
        attack_validation_rows_per_class=SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS,
    )


def load_supervised_samples_with_sizes(
    train_benign_rows: int,
    validation_benign_rows: int = SUPERVISED_VALIDATION_BENIGN_ROWS,
    attack_train_rows_per_class: int = SUPERVISED_ATTACK_TRAIN_ROWS_PER_CLASS,
    attack_validation_rows_per_class: int = SUPERVISED_ATTACK_VALIDATION_ROWS_PER_CLASS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a supervised split with the chosen benign train size and the shared test."""
    train_path, validation_path = ensure_supervised_sample_files_with_sizes(
        train_benign_rows=train_benign_rows,
        validation_benign_rows=validation_benign_rows,
        attack_train_rows_per_class=attack_train_rows_per_class,
        attack_validation_rows_per_class=attack_validation_rows_per_class,
    )
    train = pd.read_csv(train_path, low_memory=False)
    validation = pd.read_csv(validation_path, low_memory=False)
    test_frame = pd.read_csv(EVAL_SAMPLE_PATH, low_memory=False)
    return train, validation, test_frame


def sample_summary(train: pd.DataFrame, eval_frame: pd.DataFrame) -> pd.DataFrame:
    """Describe the sample composition and verify train/eval separation."""
    train_keys = train.loc[:, ["source_file", "source_row"]].drop_duplicates()
    eval_keys = eval_frame.loc[
        eval_frame["binary_label"].astype(int) == 0, ["source_file", "source_row"]
    ].drop_duplicates()
    overlap = train_keys.merge(eval_keys, how="inner", on=["source_file", "source_row"])

    attack_counts = (
        eval_frame.loc[eval_frame["binary_label"].astype(int) == 1, "coarse_label"]
        .value_counts()
        .sort_index()
    )

    rows: list[dict[str, str | int]] = [
        {"item": "train_benign_rows", "value": int(len(train))},
        {"item": "eval_benign_rows", "value": int((eval_frame["binary_label"].astype(int) == 0).sum())},
        {"item": "eval_attack_rows", "value": int((eval_frame["binary_label"].astype(int) == 1).sum())},
        {"item": "train_eval_benign_overlap", "value": int(len(overlap))},
    ]

    for label, count in attack_counts.items():
        rows.append({"item": f"eval_attack_{label}", "value": int(count)})

    return pd.DataFrame(rows)


def supervised_sample_summary(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Describe the supervised split composition and verify disjointness."""

    train_keys = frame_keys(train_frame)
    validation_keys = frame_keys(validation_frame)
    test_keys = frame_keys(test_frame)

    overlap_train_validation = len(train_keys & validation_keys)
    overlap_train_test = len(train_keys & test_keys)
    overlap_validation_test = len(validation_keys & test_keys)

    rows: list[dict[str, str | int]] = [
        {"item": "train_rows", "value": int(len(train_frame))},
        {"item": "validation_rows", "value": int(len(validation_frame))},
        {"item": "test_rows", "value": int(len(test_frame))},
        {"item": "train_benign_rows", "value": int((train_frame["binary_label"].astype(int) == 0).sum())},
        {"item": "train_attack_rows", "value": int((train_frame["binary_label"].astype(int) == 1).sum())},
        {
            "item": "validation_benign_rows",
            "value": int((validation_frame["binary_label"].astype(int) == 0).sum()),
        },
        {
            "item": "validation_attack_rows",
            "value": int((validation_frame["binary_label"].astype(int) == 1).sum()),
        },
        {"item": "test_benign_rows", "value": int((test_frame["binary_label"].astype(int) == 0).sum())},
        {"item": "test_attack_rows", "value": int((test_frame["binary_label"].astype(int) == 1).sum())},
        {"item": "train_validation_overlap", "value": int(overlap_train_validation)},
        {"item": "train_test_overlap", "value": int(overlap_train_test)},
        {"item": "validation_test_overlap", "value": int(overlap_validation_test)},
    ]

    for split_name, frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ):
        attack_counts = (
            frame.loc[frame["binary_label"].astype(int) == 1, "coarse_label"].value_counts().sort_index()
        )
        for label, count in attack_counts.items():
            rows.append({"item": f"{split_name}_attack_{label}", "value": int(count)})

    return pd.DataFrame(rows)


def prepare_features(
    train: pd.DataFrame, eval_frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Apply the preprocessing used by both program versions."""
    feature_columns: list[str] = []
    for column in train.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if column not in {"Src Port", "Dst Port", "Protocol"} and not pd.api.types.is_numeric_dtype(
            train[column]
        ):
            continue
        feature_columns.append(column)

    train_num = (
        train[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )
    eval_num = (
        eval_frame[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )

    kept_columns = [column for column in train_num.columns if train_num[column].nunique(dropna=True) > 1]
    train_num = train_num[kept_columns]
    eval_num = eval_num[kept_columns]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_num))
    x_eval = scaler.transform(imputer.transform(eval_num))
    return x_train, x_eval, kept_columns


def preprocessing_summary(train: pd.DataFrame, eval_frame: pd.DataFrame) -> pd.DataFrame:
    """Describe which columns are kept and how many rows go into the pipeline."""
    raw_feature_columns: list[str] = []
    for column in train.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if column not in {"Src Port", "Dst Port", "Protocol"} and not pd.api.types.is_numeric_dtype(
            train[column]
        ):
            continue
        raw_feature_columns.append(column)

    train_num = (
        train[raw_feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    )
    kept_columns = [column for column in train_num.columns if train_num[column].nunique(dropna=True) > 1]
    dropped_columns = [column for column in raw_feature_columns if column not in kept_columns]

    rows = [
        {"item": "train_rows", "value": int(len(train))},
        {"item": "eval_rows", "value": int(len(eval_frame))},
        {"item": "raw_feature_count", "value": int(len(raw_feature_columns))},
        {"item": "kept_feature_count", "value": int(len(kept_columns))},
        {"item": "dropped_feature_count", "value": int(len(dropped_columns))},
        {"item": "excluded_columns", "value": ", ".join(sorted(EXCLUDED_COLUMNS))},
        {"item": "kept_columns", "value": ", ".join(kept_columns)},
        {"item": "dropped_columns", "value": ", ".join(dropped_columns)},
    ]
    return pd.DataFrame(rows)


def metrics_table(eval_frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float | int]:
    """Return the standard anomaly detection metrics used in the report."""
    y_true = eval_frame["binary_label"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
