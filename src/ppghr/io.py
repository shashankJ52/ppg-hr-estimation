"""Loading and windowing for the PPG-DaLiA dataset."""

import pickle
from pathlib import Path

import numpy as np

# Sampling rates (Hz)
FS_BVP = 64
FS_ACC = 32
FS_ACT = 4
FS_ECG = 700

# Ground-truth windowing convention
WIN_SEC = 8
SHIFT_SEC = 2

ACTIVITIES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer",
    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working",
}

# repo root = two levels up from this file (src/ppghr/io.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PPG_FieldStudy"


def subject_ids(raw_dir: Path = RAW_DIR) -> list[int]:
    """Subject IDs present on disk, sorted numerically."""
    return sorted(int(p.name[1:]) for p in raw_dir.glob("S[0-9]*") if p.is_dir())


def load_subject(subject_id: int, raw_dir: Path = RAW_DIR) -> dict:
    """Load one subject's pickle. Written under Python 2, hence latin1."""
    path = raw_dir / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def wrist_streams(subject: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bvp, acc, labels, activity) — wrist inputs and ground truth only.

    Chest ECG and rpeaks are deliberately not returned: they produce the
    labels and must never reach the feature matrix.
    """
    bvp = subject["signal"]["wrist"]["BVP"][:, 0]
    acc = subject["signal"]["wrist"]["ACC"]
    labels = subject["label"]
    activity = subject["activity"][:, 0].astype(int)
    return bvp, acc, labels, activity


def n_windows(bvp: np.ndarray) -> int:
    """Number of complete ground-truth windows implied by signal length."""
    duration = len(bvp) / FS_BVP
    return int((duration - WIN_SEC) / SHIFT_SEC) + 1


def window_slice(i: int, signal: np.ndarray, fs: int) -> np.ndarray:
    """Signal segment corresponding to label index i, i.e. seconds [2i, 2i+8)."""
    start = int(i * SHIFT_SEC * fs)
    return signal[start : start + int(WIN_SEC * fs)]
