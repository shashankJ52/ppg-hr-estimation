"""Per-window feature extraction for PPG heart-rate estimation."""

import numpy as np

from .baseline import bandpass, spectrum
from .io import FS_BVP, FS_ACC, FS_ACT, window_slice, load_subject, wrist_streams

HR_LO, HR_HI = 0.5, 3.5
ACC_LO, ACC_HI = 0.3, 3.5


def _top_peaks(fb, pb, n=3):
    """Top n peaks as (freqs_bpm, powers), descending by power."""
    idx = np.argsort(pb)[-n:][::-1]
    return fb[idx] * 60, pb[idx]


def _entropy(pb):
    """Spectral entropy. High = flat spectrum = no clear periodicity."""
    p = pb / (pb.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


FEATURE_NAMES = (
    ["ppg_peak1_bpm", "ppg_peak2_bpm", "ppg_peak3_bpm"]
    + ["ppg_pow1", "ppg_pow2", "ppg_pow3"]
    + ["ppg_peak_ratio", "ppg_band_power", "ppg_entropy"]
    + [f"acc{ax}_peak_bpm" for ax in "xyz"]
    + [f"acc{ax}_peak_pow" for ax in "xyz"]
    + ["acc_total_energy"]
    + ["ppg1_to_acc_dist", "ppg2_to_acc_dist", "ppg3_to_acc_dist"]
    + ["baseline_est_bpm", "prev_baseline_bpm"]
)


def subject_features(subject_id, motion_penalty=0.5, tol=6.0,
                     sigma=20.0, prior_floor=0.15, max_jump=15.0):
    """Extract the feature matrix, labels, and activity codes for one subject.

    Filters the full recording once per channel rather than per window.
    The baseline estimate is computed in the same pass so its sequential
    state is available as a temporal feature.
    """
    subj = load_subject(subject_id)
    bvp, acc, labels, activity = wrist_streams(subj)

    bvp_f = bandpass(bvp, FS_BVP)
    acc_f = np.column_stack([bandpass(acc[:, a], FS_ACC, ACC_LO, 4.0) for a in range(3)])

    n = len(labels)
    X = np.zeros((n, len(FEATURE_NAMES)))
    act = np.zeros(n, dtype=int)
    prev = None

    for i in range(n):
        f, p = spectrum(window_slice(i, bvp_f, FS_BVP), FS_BVP)
        band = (f >= HR_LO) & (f <= HR_HI)
        fb, pb = f[band], p[band]
        pk_bpm, pk_pow = _top_peaks(fb, pb)

        acc_pk_bpm, acc_pk_pow, acc_cands = [], [], []
        for a in range(3):
            fa, pa = spectrum(window_slice(i, acc_f[:, a], FS_ACC), FS_ACC)
            ab = (fa >= ACC_LO) & (fa <= ACC_HI)
            fab, pab = fa[ab], pa[ab]
            top_f, top_p = _top_peaks(fab, pab, n=3)
            acc_pk_bpm.append(top_f[0])
            acc_pk_pow.append(top_p[0])
            acc_cands.extend(top_f)

        acc_cands = np.array(acc_cands)
        harmonics = np.concatenate([acc_cands, acc_cands / 2, acc_cands * 2])
        dists = [float(np.min(np.abs(harmonics - b))) for b in pk_bpm]

        pb_adj = pb.copy()
        near = np.min(np.abs(fb[:, None] * 60 - harmonics[None, :]), axis=1) < tol
        pb_adj[near] *= motion_penalty
        if prev is not None:
            w = np.exp(-0.5 * ((fb * 60 - prev) / sigma) ** 2)
            pb_adj = pb_adj * np.maximum(w, prior_floor)
        pick = fb[np.argmax(pb_adj)] * 60
        prev_val = prev if prev is not None else pick
        if prev is not None:
            pick = float(np.clip(pick, prev - max_jump, prev + max_jump))

        X[i] = np.concatenate([
            pk_bpm, pk_pow,
            [pk_pow[0] / (pk_pow[1] + 1e-12), pb.sum(), _entropy(pb)],
            acc_pk_bpm, acc_pk_pow,
            [np.sum(acc_pk_pow)],
            dists,
            [pick, prev_val],
        ])

        seg = np.unique(window_slice(i, activity, FS_ACT))
        act[i] = -1 if len(seg) > 1 else int(seg[0])
        prev = pick

    return X, labels, act