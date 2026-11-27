"""Tuning on the validation slice, then 5-seed training of every arm on the frozen split."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from . import config as C
from . import models as M

MODEL_DIR = C.RESULTS / "models"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------- tuning
def tune_tree(d, name, block, w):
    cols = d.block(block)
    tr, va = d.mask(C.TUNE_TRAIN_STEPS), d.mask(C.TUNE_VAL_STEPS)
    rows = []
    for params in M.TREE_GRIDS[name]:
        m = M.fit_tree(name, params, d.X[tr][:, cols], d.y[tr], w, seed=0)
        ap = average_precision_score(d.y[va], m.predict_proba(d.X[va][:, cols])[:, 1])
        rows.append({**params, "val_PR_AUC": ap})
    res = pd.DataFrame(rows)
    best = res.loc[res.val_PR_AUC.idxmax()].drop("val_PR_AUC").to_dict()
    best = {k: (None if (isinstance(v, float) and np.isnan(v)) else
                int(v) if isinstance(v, float) and v.is_integer() and k != "max_features" else v)
            for k, v in best.items()}
    return best, res


def tune_gnn(d, kind, block, w):
    cols = d.block(block)
    x, ei, ids = M.graph_tensors(d, cols, C.TUNE_TRAIN_STEPS)
    xv, eiv, idv = M.graph_tensors(d, cols, C.TUNE_VAL_STEPS)
    y_tr, y_va = d.y[ids].clip(0), d.y[idv]
    tr_idx, va_idx = np.where(d.y[ids] >= 0)[0], np.where(y_va >= 0)[0]
    rows = []
    for params in M.GNN_GRIDS[kind]:
        _, curve = M.fit_gnn(kind, params, x, ei, y_tr, tr_idx, w, 0, M.MAX_EPOCHS,
                             val_graph=(xv, eiv, y_va, va_idx))
        ep, ap = max(curve, key=lambda t: t[1])
        rows.append({**params, "epochs": ep, "val_PR_AUC": ap})
        log(f"    tune {kind}/{block} {params} -> best epoch {ep}, val PR-AUC {ap:.3f}")
    res = pd.DataFrame(rows)
    b = res.loc[res.val_PR_AUC.idxmax()]
    return {"hidden": int(b.hidden), "lr": float(b.lr), "epochs": int(b.epochs)}, res


# ------------------------------------------------------------------------ training
def run_tree(d, arm, name, params, w, seeds, sampling="weights", save_seed0=True):
    cols = d.block(M.ARMS[arm]["block"])
    tr = d.mask(C.TRAIN_STEPS)
    te_all = d.mask(C.TEST_STEPS, labelled=False)
    out = []
    for s in seeds:
        m = M.fit_tree(name, params, d.X[tr][:, cols], d.y[tr], w, s, sampling)
        p = m.predict_proba(d.X[te_all][:, cols])[:, 1]
        out.append(_pred_frame(d, te_all, p, arm, name, s, sampling))
        if s == seeds[0] and save_seed0:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(m, MODEL_DIR / f"{arm}_{name}_{sampling}_seed{s}.joblib")
    return pd.concat(out)


def run_gnn(d, arm, kind, params, w, seeds):
    cols = d.block(M.ARMS[arm]["block"])
    x, ei, ids = M.graph_tensors(d, cols, C.TRAIN_STEPS)
    xt, eit, idt = M.graph_tensors(d, cols, C.TEST_STEPS)
    tr_idx = np.where(d.y[ids] >= 0)[0]
    te_all = np.zeros(len(d.y), bool); te_all[idt] = True
    out = []
    for s in seeds:
        m, _ = M.fit_gnn(kind, params, x, ei, d.y[ids].clip(0), tr_idx, w, s, params["epochs"])
        p = M.predict_gnn(m, xt, eit)            # rows ordered as idt == np.where(te_all)
        out.append(_pred_frame(d, te_all, p, arm, kind, s, "weights"))
        if s == seeds[0]:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": m.state_dict(), "params": params, "in_dim": len(cols)},
                       MODEL_DIR / f"{arm}_{kind}_weights_seed{s}.pt")
        log(f"    {arm} {kind} seed {s} done")
    return pd.concat(out)


def _pred_frame(d, rows, p, arm, model, seed, sampling):
    return pd.DataFrame({"txId": d.txid[rows], "time_step": d.time_step[rows], "y": d.y[rows],
                         "prob": p.astype(np.float32), "arm": arm, "model": model,
                         "seed": seed, "sampling": sampling})


# ------------------------------------------------------------------------ driver
def run_all(d, seeds=C.SEEDS, do_ablation=True):
    C.TABLES.mkdir(parents=True, exist_ok=True)
    w = M.illicit_weight(d.y[d.mask(C.TRAIN_STEPS)])
    log(f"illicit class weight (shared by all arms) = {w:.3f}")
    best, tuning, preds = {}, [], []
    for arm, spec in M.ARMS.items():
        for name in M.FAMILY_MODELS[spec["family"]]:
            log(f"{arm} / {name}: tuning on steps 1-28 -> 29-34")
            if spec["family"] == "tree":
                params, res = tune_tree(d, name, spec["block"], w)
            else:
                params, res = tune_gnn(d, name, spec["block"], w)
            tuning.append(res.assign(arm=arm, model=name))
            best[f"{arm}/{name}"] = params
            log(f"  best {params}; training {len(seeds)} seeds on 1-34")
            if spec["family"] == "tree":
                preds.append(run_tree(d, arm, name, params, w, seeds))
            else:
                preds.append(run_gnn(d, arm, name, params, w, seeds))
            pd.concat(preds).to_parquet(C.RESULTS / "predictions.parquet")   # checkpoint

    if do_ablation:
        for arm in ["A1", "A2"]:
            log(f"Sampling ablation: {arm} XGBoost with NCL-SMOTE instead of class weights")
            preds.append(run_tree(d, arm, "XGBoost", best[f"{arm}/XGBoost"], w, seeds, "ncl_smote"))

    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(C.RESULTS / "predictions.parquet")
    pd.concat(tuning).to_csv(C.TABLES / "tuning_results.csv", index=False)
    json.dump({"illicit_weight": w, "seeds": list(seeds), "best_params": best},
              open(C.RESULTS / "run_config.json", "w"), indent=2, default=str)
    log("all arms finished")
    return P
