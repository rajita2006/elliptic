"""Stage 5: the four controlled arms (slide 7).

  A1  tree ensemble  on LF (93)                 -> true non-graph baseline
  A2  tree ensemble  on LF + AF (165)           -> value of precomputed 1-hop aggregates
  A3  GNN            on LF (93) + graph edges   -> learned message passing, no aggregates
  A4  GNN            on LF + AF + graph edges   -> both forms of graph signal

Held constant: frozen temporal split, the same illicit-class weight in every loss,
5 seeds, illicit-class metrics. Only the input signal differs.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph, to_undirected
from xgboost import XGBClassifier

from . import config as C

ARMS = {
    "A1": {"block": "LF", "family": "tree"},
    "A2": {"block": "ALL", "family": "tree"},
    "A3": {"block": "LF", "family": "gnn"},
    "A4": {"block": "ALL", "family": "gnn"},
}
FAMILY_MODELS = {"tree": ["XGBoost", "RandomForest"], "gnn": ["GraphSAGE", "GCN"]}


def illicit_weight(y_train):
    """One weight, computed once from the training labels, shared by every arm."""
    return float((y_train == C.LICIT).sum() / (y_train == C.ILLICIT).sum())


# =============================================================== tree ensembles
TREE_GRIDS = {
    "XGBoost": [dict(max_depth=d, n_estimators=n, learning_rate=0.1)
                for d, n in itertools.product([4, 6, 8], [200, 500])],
    "RandomForest": [dict(max_depth=d, max_features=m, n_estimators=300)
                     for d, m in itertools.product([None, 20], ["sqrt", 0.3])],
}


def make_tree(name, params, w, seed):
    if name == "XGBoost":
        return XGBClassifier(**params, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             scale_pos_weight=w, random_state=seed, n_jobs=2, eval_metric="aucpr")
    return RandomForestClassifier(**params, class_weight={C.LICIT: 1.0, C.ILLICIT: w},
                                  random_state=seed, n_jobs=2)


def ncl_smote(X, y, seed):
    """Sampling-ablation variant: SMOTE oversampling of the illicit class followed by
    Neighbourhood Cleaning Rule on the licit class. Replaces (does not add to) class weights."""
    Xs, ys = SMOTE(random_state=seed).fit_resample(X, y)
    return NeighbourhoodCleaningRule(n_jobs=2).fit_resample(Xs, ys)


def fit_tree(name, params, X, y, w, seed, sampling="weights"):
    if sampling == "ncl_smote":
        X, y = ncl_smote(X, y, seed)
        w = 1.0
    return make_tree(name, params, w, seed).fit(X, y)


# =================================================================== graph models
class GNN(torch.nn.Module):
    def __init__(self, kind, in_dim, hidden, dropout=0.3):
        super().__init__()
        Conv = SAGEConv if kind == "GraphSAGE" else GCNConv
        self.c1, self.c2 = Conv(in_dim, hidden), Conv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei):
        x = F.dropout(F.relu(self.c1(x, ei)), self.dropout, self.training)
        x = F.dropout(F.relu(self.c2(x, ei)), self.dropout, self.training)
        return self.lin(x)


GNN_GRIDS = {k: [dict(hidden=h, lr=lr) for h, lr in itertools.product([64, 128], [0.01, 0.003])]
             for k in ["GraphSAGE", "GCN"]}
MAX_EPOCHS = 200
EVAL_EVERY = 10


def graph_tensors(d, cols, steps):
    """Node features + edges for the sub-graph of the given time steps.

    No edge crosses a time step, so the sub-graph of a period is exactly that period's part of
    the full graph: training never sees test-period nodes, and this is ~3x cheaper than
    propagating over all 203,769 nodes. Edges are used in both directions, so a transaction
    learns from the transactions that funded it and from those it funded.
    Returns (x, edge_index, node_ids) where node_ids maps sub-graph rows back to rows of d.
    """
    node_ids = np.where(d.mask(steps, labelled=False))[0]
    ei, _ = subgraph(torch.from_numpy(node_ids), torch.from_numpy(d.edge_index),
                     relabel_nodes=True, num_nodes=len(d.y))
    x = torch.from_numpy(d.X[node_ids][:, cols])
    return x, to_undirected(ei, num_nodes=len(node_ids)), node_ids


def fit_gnn(kind, params, x, ei, y, train_idx, w, seed, epochs, val_graph=None):
    """Full-batch training on the training-period sub-graph; loss only on labelled nodes
    (train_idx, sub-graph row numbers). If val_graph=(x_val, ei_val, y_val, val_idx) is given,
    validation PR-AUC is recorded every EVAL_EVERY epochs (tuning only)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GNN(kind, x.shape[1], params["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=5e-4)
    yt = torch.from_numpy(y)
    tr = torch.from_numpy(train_idx)
    cw = torch.tensor([1.0, w])            # index 0 = licit, 1 = illicit
    curve = []
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x, ei)[tr], yt[tr], weight=cw)
        loss.backward(); opt.step()
        if val_graph is not None and ep % EVAL_EVERY == 0:
            xv, eiv, yv, vi = val_graph
            curve.append((ep, average_precision_score(yv[vi], predict_gnn(model, xv, eiv)[vi])))
    return model, curve


@torch.no_grad()
def predict_gnn(model, x, ei):
    model.eval()
    return torch.softmax(model(x, ei), 1)[:, 1].numpy()
