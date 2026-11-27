"""Slide 8, with real numbers: trace one transaction through every stage of the system."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats

from . import config as C
from . import models as M
from .elliptical import MahalanobisModel


def pick_example(d, step=38, label=C.ILLICIT, caught=True, arm="A2", model="RandomForest"):
    """The best-connected illicit transaction at `step` that the given arm (seed 0)
    did (caught=True) or did not (caught=False) flag."""
    P = pd.read_parquet(C.RESULTS / "predictions.parquet",
                        filters=[("arm", "==", arm), ("model", "==", model), ("seed", "==", 0),
                                 ("sampling", "==", "weights"), ("time_step", "==", step)])
    hit = set(P.txId[(P.prob >= C.THRESHOLD) == caught])
    deg = np.bincount(d.edge_index.ravel(), minlength=len(d.y))
    cand = np.where((d.time_step == step) & (d.y == label))[0]
    cand = cand[np.argsort(-deg[cand], kind="stable")]
    return next(int(t) for t in d.txid[cand] if t in hit)


def trace(d, txid, rep):
    i = int(np.where(d.txid == txid)[0][0])
    out_e = int((d.edge_index[0] == i).sum()); in_e = int((d.edge_index[1] == i).sum())
    split = "train" if d.time_step[i] in C.TRAIN_STEPS else "test"
    rows = [("Input", f"txId {txid} at time step {d.time_step[i]}, 165 feature values, {in_e} incoming and "
                      f"{out_e} outgoing edges. True label: {['licit', 'illicit'][d.y[i]] if d.y[i] >= 0 else 'unknown'}.",
             "Raw node record + edges")]
    rows.append(("Preprocessing", f"Integrity checks pass. Raw LF_1 = {d.X_raw[i, 0]:.4f} becomes {d.X[i, 0]:.4f} after "
                                  f"scaling with training-period (t1-34) statistics. Assigned to the {split} set.",
                 f"Scaled vector, {split} set"))
    rows.append(("Feature block separation", f"LF vector of {C.N_LOCAL} values (first three: "
                 f"{[round(float(v), 3) for v in d.X[i, :3]]}), AF vector of {C.N_AGG} values.", "LF (93), AF (72)"))
    rows.append(("Representation", f"Row {i} of X (203,769 x 165) for the tree arms; node {i} of G with "
                 f"{in_e + out_e} undirected neighbours for the graph arms.", "Row in X, node in G"))

    tr_licit = d.mask(C.TRAIN_STEPS) & (d.y == C.LICIT)
    mm = MahalanobisModel().fit(d.X[tr_licit][:, d.lf_idx])
    d2 = float(mm.d2(d.X[i:i + 1, d.lf_idx])[0]); thr = stats.chi2(mm.df).ppf(.975)
    rows.append(("Elliptical check", f"Squared Mahalanobis distance {d2:.1f} vs chi-square({mm.df}) 97.5% threshold {thr:.1f}.",
                 "Flagged as distant" if d2 > thr else "Inside the licit ellipsoid — the elliptical rule would miss it"))

    probs = {}
    for arm, spec in M.ARMS.items():
        model = rep[arm]
        cols = d.block(spec["block"])
        if spec["family"] == "tree":
            m = joblib.load(C.RESULTS / "models" / f"{arm}_{model}_weights_seed0.joblib")
            probs[arm] = float(m.predict_proba(d.X[i:i + 1, cols])[0, 1])
        else:
            ck = torch.load(C.RESULTS / "models" / f"{arm}_{model}_weights_seed0.pt", weights_only=False)
            net = M.GNN(model, ck["in_dim"], ck["params"]["hidden"]); net.load_state_dict(ck["state_dict"])
            x, ei, ids = M.graph_tensors(d, cols, [int(d.time_step[i])])
            probs[arm] = float(M.predict_gnn(net, x, ei)[np.where(ids == i)[0][0]])
    rows.append(("Detection model", ", ".join(f"{a} ({rep[a]}) = {p:.3f}" for a, p in probs.items()), "Four probability scores"))
    votes = [a for a, p in probs.items() if p >= C.THRESHOLD]
    rows.append(("Classification", f"Threshold {C.THRESHOLD}: predicted ILLICIT by {votes or 'no arm'}.",
                 f"{len(votes)}/4 arms predict illicit"))
    rows.append(("Evaluation", f"Added to the confusion matrix for time step {d.time_step[i]} "
                 f"({'pre' if d.time_step[i] < C.CHANGE_POINT else 'post'}-change window).",
                 f"One entry in the t={d.time_step[i]} metric set"))
    return pd.DataFrame(rows, columns=["stage", "what happens to this transaction", "state after the stage"])
