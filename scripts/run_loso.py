#!/usr/bin/env python
"""Reproduce the headline LOSO results.

Usage: python scripts/run_loso.py
"""
import json
import pickle
import time

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor

from ppghr.features import subject_features, FEATURE_NAMES
from ppghr.io import subject_ids, PROJECT_ROOT

CACHE = PROJECT_ROOT / "data" / "processed" / "features.pkl"
OUT = PROJECT_ROOT / "results" / "metrics"


def build_features():
    if CACHE.exists():
        print(f"loading cached features from {CACHE.relative_to(PROJECT_ROOT)}")
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        return d["X"], d["y"], d["act"], d["groups"]

    Xs, ys, acts, gs = [], [], [], []
    for sid in subject_ids():
        Xi, yi, ai = subject_features(sid)
        Xs.append(Xi); ys.append(yi); acts.append(ai)
        gs.append(np.full(len(yi), sid))
        print(f"  S{sid:<2} {len(yi):5d} windows")

    X, y = np.vstack(Xs), np.concatenate(ys)
    act, groups = np.concatenate(acts), np.concatenate(gs)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"X": X, "y": y, "act": act, "groups": groups}, f)
    return X, y, act, groups


def loso(X, y, groups, cols=None):
    cols = slice(None) if cols is None else cols
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         objective="reg:absoluteerror", n_jobs=-1, random_state=0)
        m.fit(X[tr][:, cols], y[tr])
        pred[te] = m.predict(X[te][:, cols])
    return pred


def main():
    t0 = time.time()
    X, y, act, groups = build_features()
    print(f"{X.shape[0]} windows, {len(np.unique(groups))} subjects, {X.shape[1]} features\n")

    names = list(FEATURE_NAMES)
    acc_idx = [i for i, n in enumerate(names) if n.startswith("acc")]
    ppg_idx = [i for i, n in enumerate(names) if n.startswith("ppg")]
    nb_idx = [i for i, n in enumerate(names)
              if n not in ("baseline_est_bpm", "prev_baseline_bpm")]

    mae = lambda p: float(np.abs(p - y).mean())
    res = {
        "constant_predictor": float(np.abs(y - y.mean()).mean()),
        "spectral_baseline": mae(X[:, names.index("baseline_est_bpm")]),
        "xgb_acc_only": mae(loso(X, y, groups, acc_idx)),
        "xgb_ppg_only": mae(loso(X, y, groups, ppg_idx)),
        "xgb_no_baseline_feats": mae(loso(X, y, groups, nb_idx)),
        "xgb_full": mae(loso(X, y, groups)),
        "published_classical": 11.06,
        "published_cnn": 7.65,
        "n_windows": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "loso_results.json", "w") as f:
        json.dump(res, f, indent=2)

    print(json.dumps(res, indent=2))
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()