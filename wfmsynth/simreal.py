"""
wfmsynth.simreal — a sim-to-real separability harness.

Given a set of synthetic waveforms and a set of real captures, the question that tells you
whether any of the realism work matters is: can a trivial classifier separate them, and if
so, *which feature does it use?* A separating feature names the missing physics instead of
leaving you to guess, and turns a realism backlog into an evidence-ordered work queue.

This computes a common, cheap feature vector per waveform and, for each feature, the
rank-based AUC between the two sets (a single-feature-threshold "trivial classifier"). The
top feature is the one a model would exploit to tell synthetic from real. numpy/scipy only —
no sklearn; the trivial classifier is the point.

    rep = separability(synthetic_set, real_set, grid)
    rep["best_feature"], rep["best_auc"]      # e.g. ("hf_fraction", 0.98) -> your HF content is wrong
    rep["auc"]                                # AUC per feature, the whole diagnosis
"""
from __future__ import annotations

import numpy as np


def feature_vector(x, grid=None):
    """A small, cheap, general feature vector for a waveform — amplitude, shape, and
    spectral descriptors that a classifier could use to tell two populations apart."""
    x = np.asarray(x, float)
    xc = x - x.mean()
    rms = np.sqrt(np.mean(xc ** 2)) + 1e-12
    X = np.abs(np.fft.rfft(xc)) ** 2
    f = np.arange(len(X))
    total = X.sum() + 1e-12
    return {
        "rms": rms,
        "ptp": float(np.ptp(x)),
        "crest": float(np.ptp(x) / rms),
        "kurtosis": float(np.mean(xc ** 4) / (np.mean(xc ** 2) ** 2)),
        "skew": float(np.mean(xc ** 3) / (rms ** 3)),
        "spectral_centroid": float((f * X).sum() / total),
        "hf_fraction": float(X[len(X) // 2:].sum() / total),
        "zero_cross_rate": float(np.mean(np.abs(np.diff(np.sign(xc))) > 0)),
        "ac_lag1": float(np.sum(xc[1:] * xc[:-1]) / np.sum(xc ** 2)),
    }


def _auc(scores, labels):
    """Rank-based (Mann-Whitney) AUC, folded to [0.5, 1] so it measures separability in
    either direction. 0.5 = indistinguishable, 1.0 = perfectly separable on this score."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    auc = (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return max(auc, 1.0 - auc)


def separability(set_a, set_b, grid=None, features=None):
    """Can a trivial classifier tell ``set_a`` (e.g. synthetic) from ``set_b`` (e.g. real)?
    Computes ``feature_vector`` for every waveform and the per-feature separability AUC.
    Returns ``{"auc": {feat: auc}, "best_feature", "best_auc", "n_a", "n_b"}``. A ``best_auc``
    near 0.5 means the two sets are indistinguishable on these features (good); a high one
    names the feature — hence the missing physics — a model would key on."""
    fa = [feature_vector(x, grid) for x in set_a]
    fb = [feature_vector(x, grid) for x in set_b]
    names = features or list(fa[0])
    labels = np.array([0] * len(fa) + [1] * len(fb))
    auc = {}
    for name in names:
        scores = np.array([d[name] for d in fa] + [d[name] for d in fb])
        if np.ptp(scores) == 0:
            auc[name] = 0.5
        else:
            auc[name] = _auc(scores, labels)
    best = max(auc, key=auc.get)
    return {"auc": auc, "best_feature": best, "best_auc": auc[best],
            "n_a": len(fa), "n_b": len(fb)}
