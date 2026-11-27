"""Slide 6: test (not assume) whether licit transactions are elliptically distributed.

If licit points formed one ellipsoid, squared Mahalanobis distances d^2 from the licit centroid
would follow chi-square(p). We fit mean/covariance on licit TRAINING points only and check that,
then use d^2 as an unsupervised anomaly score to see whether "far from the ellipsoid" means illicit.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from . import config as C


class MahalanobisModel:
    def __init__(self, rel_tol=1e-6):
        self.rel_tol = rel_tol

    def fit(self, X):
        self.mu = X.mean(0)
        S = np.cov(X - self.mu, rowvar=False)
        w, V = np.linalg.eigh(S)
        keep = w > self.rel_tol * w.max()          # drop numerically-zero directions
        self.df = int(keep.sum())
        self.W = V[:, keep] / np.sqrt(w[keep])      # whitening map on the non-degenerate subspace
        self.p = X.shape[1]
        return self

    def d2(self, X):
        Z = (X - self.mu) @ self.W
        return (Z ** 2).sum(1)


def run(d, block="LF", out_dir=C.FIGURES):
    cols = d.block(block)
    tr_licit = d.mask(C.TRAIN_STEPS) & (d.y == C.LICIT)
    lab = d.y >= 0
    m = MahalanobisModel().fit(d.X[tr_licit][:, cols])
    d2 = np.full(len(d.y), np.nan)
    d2[lab] = m.d2(d.X[lab][:, cols])
    p = m.df

    # ---- goodness of fit of d^2 ~ chi2(p) on the licit training points -----------
    x = d2[tr_licit]
    ks = stats.kstest(x, stats.chi2(p).cdf)
    q95, q975 = stats.chi2(p).ppf([0.95, 0.975])
    mardia_b2 = float(np.mean(x ** 2))               # Mardia kurtosis, = p(p+2) under normality
    gof = pd.DataFrame([
        ("Feature block", block), ("Dimensions in block", m.p), ("Effective rank (chi-square df)", p),
        ("Licit training points", int(tr_licit.sum())),
        ("Mean d^2 (expected = df)", round(float(x.mean()), 1)),
        ("Median d^2 (expected)", f"{np.median(x):.1f} ({stats.chi2(p).median():.1f})"),
        ("Share above chi2 95% quantile (expected 5%)", f"{(x > q95).mean():.1%}"),
        ("Mardia kurtosis b2 (expected p(p+2))", f"{mardia_b2:,.0f} ({p * (p + 2):,})"),
        ("KS statistic vs chi2(df)", round(ks.statistic, 3)),
        ("KS p-value", f"{ks.pvalue:.2e}"),
        ("Verdict", "elliptical assumption REJECTED" if ks.pvalue < 0.01 else "not rejected"),
    ], columns=["quantity", "value"])

    # ---- d^2 as an unsupervised detector on the test period ------------------------
    te = d.mask(C.TEST_STEPS)
    yt, st = d.y[te], d2[te]
    pred = (st > q975).astype(int)
    det = pd.DataFrame([{
        "rule": f"flag if d^2 > chi2_{p}(0.975) = {q975:.1f}",
        "flagged_%": round(pred.mean() * 100, 1),
        "precision": precision_score(yt, pred, zero_division=0),
        "recall": recall_score(yt, pred),
        "illicit_F1": f1_score(yt, pred),
        "PR_AUC (score=d^2)": average_precision_score(yt, st),
        "PR_AUC of random ranking": yt.mean(),
    }]).round(3)

    _plots(d, d2, m, tr_licit, q975, out_dir)
    return {"gof": gof, "detector": det, "d2": d2, "model": m}


def _plots(d, d2, m, tr_licit, q975, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = m.df
    tr = d.mask(C.TRAIN_STEPS)
    rng = np.random.default_rng(0)

    # (a) PCA scatter of all 165 scaled features with the licit 95 % covariance ellipse
    pca = PCA(2, random_state=0).fit(d.X[tr])
    Z = pca.transform(d.X[tr])
    yy = d.y[tr]
    zl = Z[yy == C.LICIT]
    mu, S = zl.mean(0), np.cov(zl, rowvar=False)
    w, V = np.linalg.eigh(S)
    k = np.sqrt(stats.chi2(2).ppf(0.95))
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    li = rng.choice(np.where(yy == C.LICIT)[0], 6000, replace=False)
    il = np.where(yy == C.ILLICIT)[0]
    ax.scatter(Z[li, 0], Z[li, 1], s=5, c="#4C78A8", alpha=.35, label="licit (6k sample)")
    ax.scatter(Z[il, 0], Z[il, 1], s=9, c="#E45756", marker="x", alpha=.7, label="illicit")
    ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang,
                         fill=False, lw=2, color="black", label="licit 95% ellipse"))
    inside = (((Z[il] - mu) @ np.linalg.inv(S)) * (Z[il] - mu)).sum(1) <= k ** 2
    ax.set(title=f"PCA, train period: {inside.mean():.0%} of illicit points lie inside the licit ellipse",
           xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    lo, hi = np.percentile(Z, [0.5, 99.5], axis=0)
    ax.set_xlim(lo[0] - 1, hi[0] + 1); ax.set_ylim(lo[1] - 1, hi[1] + 1)
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(out_dir / "elliptical_pca_ellipse.png", dpi=150); plt.close(fig)

    # (b) chi-square QQ plot
    x = np.sort(d2[tr_licit])
    th = stats.chi2(p).ppf((np.arange(1, len(x) + 1) - .5) / len(x))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.loglog(th, x, ".", ms=2, color="#4C78A8", label="licit training points")
    lim = [th.min(), max(th.max(), x.max())]
    ax.loglog(lim, lim, "k--", lw=1, label="y = x (elliptical)")
    ax.set(title=f"Chi-square Q-Q plot, df = {p}", xlabel="theoretical chi-square quantile",
           ylabel="observed squared Mahalanobis distance"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "elliptical_chi2_qq.png", dpi=150); plt.close(fig)

    # (c) per-class histogram of d^2 (test period, log scale)
    te = d.mask(C.TEST_STEPS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo, hi = np.nanpercentile(d2[te], [0.1, 99.9])
    bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 50)
    for lab, col, name in [(C.LICIT, "#4C78A8", "licit"), (C.ILLICIT, "#E45756", "illicit")]:
        v = np.clip(d2[te & (d.y == lab)], bins[0], bins[-1])
        ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="stepfilled",
                alpha=.45, color=col, ec=col, lw=1.2, label=f"{name} (n={len(v):,})")
    ax.axvline(q975, color="k", ls="--", lw=1, label=f"chi2 97.5% threshold ({q975:.0f})")
    ax.set(xscale="log", title="Squared Mahalanobis distance by class (test period)",
           xlabel="d^2 from licit centroid", ylabel="share of class"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "elliptical_distance_hist.png", dpi=150); plt.close(fig)
