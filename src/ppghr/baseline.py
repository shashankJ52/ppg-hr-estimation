"""Spectral heart-rate baseline with accelerometer-based motion suppression."""

import numpy as np
from scipy.signal import butter, filtfilt, welch

from .io import FS_BVP, FS_ACC, window_slice


def bandpass(x, fs, low=0.5, high=4.0, order=3):
    """Zero-phase Butterworth bandpass. Applied to the full recording, not per window."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def spectrum(x, fs, nfft=2048):
    """Welch PSD, zero-padded for finer peak localisation."""
    f, p = welch(x, fs=fs, nperseg=len(x), nfft=nfft, detrend="constant")
    return f, p


def acc_motion_peaks(i, acc, lo=0.3, hi=3.5, n=3):
    """Top n motion frequencies per axis, in bpm.

    Uses individual axes rather than vector magnitude: the norm is
    nonlinear and doubles the frequency content, hiding the true
    arm-swing fundamental.
    """
    a = window_slice(i, acc, FS_ACC)
    out = []
    for ax in range(3):
        f, p = spectrum(bandpass(a[:, ax], FS_ACC, 0.3, 4.0), FS_ACC)
        band = (f >= lo) & (f <= hi)
        fb, pb = f[band], p[band]
        out.extend(fb[np.argsort(pb)[-n:]] * 60)
    return np.array(out)


def estimate_hr(bvp_filt, acc, n_windows, lo=0.5, hi=3.5,
                motion_penalty=0.5, tol=6.0, sigma=20.0,
                prior_floor=0.15, max_jump=15.0):
    """Sequential spectral HR estimate.

    Motion peaks and their half/double harmonics are attenuated rather
    than removed, since HR and cadence genuinely coincide in ~11% of
    windows. The continuity prior has a floor and a slew limit so a
    transient lock-on cannot become permanent.
    """
    est = np.zeros(n_windows)
    prev = None
    for i in range(n_windows):
        f, p = spectrum(window_slice(i, bvp_filt, FS_BVP), FS_BVP)
        band = (f >= lo) & (f <= hi)
        fb, pb = f[band] * 60, p[band].copy()

        pk = acc_motion_peaks(i, acc)
        cands = np.concatenate([pk, pk / 2, pk * 2])
        pb[np.min(np.abs(fb[:, None] - cands[None, :]), axis=1) < tol] *= motion_penalty

        if prev is not None:
            w = np.exp(-0.5 * ((fb - prev) / sigma) ** 2)
            pb = pb * np.maximum(w, prior_floor)

        pick = fb[np.argmax(pb)]
        if prev is not None:
            pick = np.clip(pick, prev - max_jump, prev + max_jump)
        est[i] = prev = pick
    return est