"""Stages 6-7: classification, per-time-step evaluation, drift analysis, attribution, alert list."""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from . import config as C
from . import models as M

KEYS = ["arm", "model", "sampling"]
ARM_COLORS = {"A1": "#4C78A8", "A2": "#F58518", "A3": "#54A24B", "A4": "#B279A2"}
ARM_LABEL = {"A1": "A1 tree · LF", "A2": "A2 tree · LF+AF", "A3": "A3 GNN · LF+edges",
             "A4": "A4 GNN · LF+AF+edges"}


# ------------------------------------------------------------------------- metrics
def _metrics(y, p, thr=C.THRESHOLD):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0),
                f1=f1_score(y, yhat, zero_division=0),
                pr_auc=average_precision_score(y, p) if both else np.nan,
                roc_auc=roc_auc_score(y, p) if both else np.nan,
                accuracy=accuracy_score(y, yhat), tp=tp, fp=fp, tn=tn, fn=fn, n_illicit=int(y.sum()), n=len(y))


def per_step_metrics(P):
    L = P[P.y >= 0]
    rows = [{**dict(zip(KEYS + ["seed", "time_step"], k)), **_metrics(g.y.values, g.prob.values)}
            for k, g in L.groupby(KEYS + ["seed", "time_step"])]
    return pd.DataFrame(rows)


def pooled_metrics(P):
    L = P[P.y >= 0]
    rows = [{**dict(zip(KEYS + ["seed"], k)), **_metrics(g.y.values, g.prob.values)}
            for k, g in L.groupby(KEYS + ["seed"])]
    return pd.DataFrame(rows)


def ms(s, nd=3):
    return f"{s.mean():.{nd}f} ± {s.std(ddof=1) if len(s) > 1 else 0:.{nd}f}"


def summary_table(pooled, steps):
    pre = steps[steps.time_step.isin(C.PRE_WINDOW)].groupby(KEYS + ["seed"]).f1.mean().rename("pre")
    post = steps[steps.time_step.isin(C.POST_WINDOW)].groupby(KEYS + ["seed"]).f1.mean().rename("post")
    step_mean = steps.groupby(KEYS + ["seed"]).f1.mean().rename("step_mean")
    j = pooled.set_index(KEYS + ["seed"]).join([pre, post, step_mean]).reset_index()
    j["delta"] = j.post - j.pre
    out = []
    for k, g in j.groupby(KEYS):
        out.append({**dict(zip(KEYS, k)),
                    "illicit_F1 (pooled)": ms(g.f1), "precision": ms(g.precision), "recall": ms(g.recall),
                    "PR_AUC": ms(g.pr_auc), "ROC_AUC": ms(g.roc_auc), "accuracy": ms(g.accuracy),
                    "F1 mean over steps": ms(g.step_mean), "F1 pre t35-42": ms(g.pre),
                    "F1 post t43-49": ms(g.post), "post - pre": ms(g.delta),
                    "_f1": g.f1.mean(), "_pr": g.pr_auc.mean(), "_pre": g.pre.mean(), "_post": g.post.mean()})
    return pd.DataFrame(out).sort_values(["sampling", "arm", "model"], ascending=[False, True, True])


def representative_models():
    """One model per arm, chosen on the VALIDATION slice (never on the test period)."""
    t = pd.read_csv(C.TABLES / "tuning_results.csv")
    best = t.loc[t.groupby("arm").val_PR_AUC.idxmax(), ["arm", "model", "val_PR_AUC"]]
    return dict(zip(best.arm, best.model)), best


# ------------------------------------------------------------------------- figures
def fig_drift(steps, rep, out):
    S = steps[steps.sampling == "weights"]
    counts = S.groupby("time_step").n_illicit.first()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, mode in zip(axes, ["rep", "all"]):
        ax2 = ax.twinx()
        ax2.bar(counts.index, counts.values, color="#BBBBBB", alpha=.35, width=.8)
        ax2.set_ylabel("illicit transactions in step", color="#888888"); ax2.set_ylim(0, counts.max() * 3)
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        for (arm, model), g in S.groupby(["arm", "model"]):
            if mode == "rep" and rep[arm] != model:
                continue
            a = g.groupby("time_step").f1.agg(["mean", "std"])
            ls = "-" if mode == "rep" or model in ("XGBoost", "GraphSAGE") else "--"
            lbl = f"{ARM_LABEL[arm]} ({model})" if mode == "rep" else f"{arm} {model}"
            ax.plot(a.index, a["mean"], ls, marker="o", ms=4, color=ARM_COLORS[arm], label=lbl)
            if mode == "rep":
                ax.fill_between(a.index, a["mean"] - a["std"], a["mean"] + a["std"], color=ARM_COLORS[arm], alpha=.15)
        ax.axvline(C.CHANGE_POINT - .5, color="crimson", lw=1.5, ls=":")
        ax.text(C.CHANGE_POINT - .4, 0.98, " dark market\n shutdown (t=43)", color="crimson", va="top", fontsize=9)
        ax.axhline(C.WEBER_RF_F1, color="k", lw=1, ls="--")
        ax.text(49.4, C.WEBER_RF_F1 + .015, "Weber et al. 2019 avg F1 = 0.788", fontsize=8, ha="right")
        ax.set(xlabel="test time step", ylim=(0, 1.02), xticks=list(C.TEST_STEPS),
               title="Representative model per arm (mean ± sd, 5 seeds)" if mode == "rep" else "All eight arm/model combinations")
        ax.legend(loc="center right", bbox_to_anchor=(1, .52), fontsize=8, framealpha=.95)
    axes[0].set_ylabel("illicit-class F1 (threshold 0.5)")
    fig.tight_layout(); fig.savefig(out / "drift_curve_illicit_f1.png", dpi=150); plt.close(fig)


def fig_prepost(summary, out):
    s = summary[summary.sampling == "weights"].copy()
    s["name"] = s.arm + "\n" + s.model
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - .2, s._pre, .4, label="pre-change t35-42", color="#4C78A8")
    ax.bar(x + .2, s._post, .4, label="post-change t43-49", color="#E45756")
    for i, (a, b) in enumerate(zip(s._pre, s._post)):
        ax.text(i, max(a, b) + .02, f"Δ {b - a:+.2f}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=s.name, ylabel="mean illicit F1 over window", ylim=(0, 1.05),
           title="Degradation after the dark-market shutdown")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "pre_vs_post_f1.png", dpi=150); plt.close(fig)


def fig_pr(P, rep, out):
    L = P[(P.y >= 0) & (P.sampling == "weights")]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for arm, model in rep.items():
        g = L[(L.arm == arm) & (L.model == model)].groupby("txId").agg(y=("y", "first"), p=("prob", "mean"))
        pr, rc, _ = precision_recall_curve(g.y, g.p)
        ax.plot(rc, pr, color=ARM_COLORS[arm], label=f"{ARM_LABEL[arm]} ({model}) AP={average_precision_score(g.y, g.p):.3f}")
    ax.axhline(L.groupby("txId").y.first().mean(), color="grey", ls=":", label="random")
    ax.set(xlabel="recall (illicit)", ylabel="precision (illicit)", title="Precision-recall, test period (seed-averaged scores)")
    ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig(out / "pr_curves.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------- feature-block attribution
def _block_share(attr_abs, d, rows, lf_n=C.N_LOCAL):
    """attr_abs: (n_rows, 165) absolute attributions. Returns LF/AF share per window."""
    ts = d.time_step[rows]
    out = []
    for win, steps in [("pre t35-42", C.PRE_WINDOW), ("post t43-49", C.POST_WINDOW)]:
        m = np.isin(ts, list(steps))
        for cls, cm in [("all", np.ones(len(rows), bool)), ("illicit", d.y[rows] == 1)]:
            a = attr_abs[m & cm]
            lf, af = a[:, :lf_n].sum(), a[:, lf_n:].sum()
            out.append({"window": win, "rows": cls, "LF_share": lf / (lf + af), "AF_share": af / (lf + af), "n": int((m & cm).sum())})
    return pd.DataFrame(out)


def shap_attribution(d):
    path = C.RESULTS / "models" / "A2_XGBoost_weights_seed0.joblib"
    m = joblib.load(path)
    rows = np.where(d.mask(C.TEST_STEPS))[0]
    sv = shap.TreeExplainer(m).shap_values(d.X[rows])
    A = np.abs(sv)
    share = _block_share(A, d, rows).assign(model="A2 XGBoost (TreeSHAP)")
    names = [f"LF_{i}" for i in range(1, C.N_LOCAL + 1)] + [f"AF_{i}" for i in range(1, C.N_AGG + 1)]
    top = []
    for win, steps in [("pre t35-42", C.PRE_WINDOW), ("post t43-49", C.POST_WINDOW)]:
        mm = np.isin(d.time_step[rows], list(steps))
        imp = pd.Series(A[mm].mean(0), index=names).sort_values(ascending=False).head(10)
        top += [{"window": win, "rank": i + 1, "feature": f, "mean_abs_shap": v} for i, (f, v) in enumerate(imp.items())]
    return share, pd.DataFrame(top)


def gnn_attribution(d, kind):
    """Gradient x input on the A4 GNN (all 165 node features), summed per block."""
    ck = torch.load(C.RESULTS / "models" / f"A4_{kind}_weights_seed0.pt", weights_only=False)
    net = M.GNN(kind, ck["in_dim"], ck["params"]["hidden"]); net.load_state_dict(ck["state_dict"]); net.eval()
    x, ei, ids = M.graph_tensors(d, d.block("ALL"), C.TEST_STEPS)
    x = x.clone().requires_grad_(True)
    logit = net(x, ei)
    lab = np.where(d.y[ids] >= 0)[0]
    (logit[lab, 1] - logit[lab, 0]).sum().backward()
    A = (x.grad * x).abs().detach().numpy()[lab]
    return _block_share(A, d, ids[lab]).assign(model=f"A4 {kind} (gradient x input)")


def fig_attribution(att, out):
    a = att[att.rows == "all"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = [f"{m}\n{w}" for m, w in zip(a.model, a.window)]
    ax.barh(labels, a.LF_share, color="#4C78A8", label="local features (LF)")
    ax.barh(labels, a.AF_share, left=a.LF_share, color="#F58518", label="aggregated features (AF)")
    for i, v in enumerate(a.LF_share):
        ax.text(v / 2, i, f"{v:.0%}", va="center", ha="center", color="white", fontsize=9)
        ax.text(v + (1 - v) / 2, i, f"{1 - v:.0%}", va="center", ha="center", color="white", fontsize=9)
    ax.set(xlim=(0, 1), xlabel="share of total attribution mass", title="Which feature block the model relies on, before vs after t=43")
    ax.invert_yaxis(); ax.legend(loc="lower center", bbox_to_anchor=(.5, -.42), ncol=2)
    fig.tight_layout(); fig.savefig(out / "feature_block_attribution.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- alerts
def alert_list(P, arm, model):
    g = P[(P.arm == arm) & (P.model == model) & (P.sampling == "weights")]
    s = g.groupby("txId").agg(time_step=("time_step", "first"), y=("y", "first"),
                              risk_score=("prob", "mean"), score_sd=("prob", "std")).reset_index()
    s["true_label"] = s.y.map({1: "illicit", 0: "licit", -1: "unknown"})
    s["predicted"] = np.where(s.risk_score >= C.THRESHOLD, "illicit", "licit")
    s = s.sort_values("risk_score", ascending=False).reset_index(drop=True)
    s.insert(0, "rank", np.arange(1, len(s) + 1))
    lab = s[s.y >= 0].copy()
    lab.insert(0, "rank_labelled", np.arange(1, len(lab) + 1))
    lab["cum_precision"] = (lab.y == 1).cumsum() / lab.rank_labelled
    lab["cum_recall"] = (lab.y == 1).cumsum() / (lab.y == 1).sum()
    cols = ["time_step", "risk_score", "score_sd", "predicted", "true_label"]
    return (lab[["rank_labelled", "txId"] + cols + ["cum_precision", "cum_recall"]],
            s[["rank", "txId"] + cols])


# ---------------------------------------------------------------------------- driver
def run(d):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(C.RESULTS / "predictions.parquet")
    steps, pooled = per_step_metrics(P), pooled_metrics(P)
    steps.to_csv(C.TABLES / "per_step_metrics_all_seeds.csv", index=False)
    summary = summary_table(pooled, steps)
    summary.drop(columns=[c for c in summary if c.startswith("_")]).to_csv(C.TABLES / "main_results.csv", index=False)

    agg = steps.groupby(KEYS + ["time_step"]).agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                                                 precision=("precision", "mean"), recall=("recall", "mean"),
                                                 tp=("tp", "mean"), fp=("fp", "mean"), fn=("fn", "mean"), tn=("tn", "mean"),
                                                 n_illicit=("n_illicit", "first")).reset_index()
    agg.round(4).to_csv(C.TABLES / "per_step_f1_and_confusion.csv", index=False)
    wide = agg[agg.sampling == "weights"].pivot_table(index="time_step", columns=["arm", "model"], values="f1_mean")
    wide.round(3).to_csv(C.TABLES / "per_step_f1_wide.csv")

    rep, rep_tbl = representative_models()
    rep_tbl.to_csv(C.TABLES / "representative_model_per_arm.csv", index=False)
    fig_drift(steps, rep, C.FIGURES); fig_prepost(summary, C.FIGURES); fig_pr(P, rep, C.FIGURES)

    shap_share, top = shap_attribution(d)
    att = pd.concat([shap_share, gnn_attribution(d, rep["A4"])])
    att.round(4).to_csv(C.TABLES / "feature_block_attribution.csv", index=False)
    top.to_csv(C.TABLES / "top_shap_features_pre_post.csv", index=False)
    fig_attribution(att, C.FIGURES)

    # alert list from the model with the best VALIDATION PR-AUC overall
    best = rep_tbl.sort_values("val_PR_AUC", ascending=False).iloc[0]
    lab, allscores = alert_list(P, best.arm, best.model)
    lab.to_csv(C.RESULTS / f"alert_list_{best.arm}_{best.model}.csv", index=False)
    allscores.to_csv(C.RESULTS / f"risk_scores_test_period_{best.arm}_{best.model}.csv", index=False)
    json.dump({"alert_model": f"{best.arm}/{best.model}", "representative": rep},
              open(C.RESULTS / "analysis_meta.json", "w"), indent=2)
    print(summary.drop(columns=[c for c in summary if c.startswith("_")]).to_string(index=False))
    return summary
