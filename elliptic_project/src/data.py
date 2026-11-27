"""Stage 1-4 of the pipeline: load, validate, map labels, split, scale, separate feature blocks.

Accepts either release of the data, placed in data/raw/:
  * Kaggle  (ellipticco/elliptic-data-set): elliptic_txs_features.csv (no header),
                                             elliptic_txs_classes.csv, elliptic_txs_edgelist.csv
  * Elliptic++ (git-disl/EllipticPlusPlus):  txs_features.csv, txs_classes.csv, txs_edgelist.csv
    Elliptic++ contains the same 203,769 transactions and the same 166 original columns plus
    17 extra augmented columns, which are dropped here so both sources give identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C

LF_COLS = [f"LF_{i}" for i in range(1, C.N_LOCAL + 1)]
AF_COLS = [f"AF_{i}" for i in range(1, C.N_AGG + 1)]
FEATURE_COLS = LF_COLS + AF_COLS
CACHE = C.DATA_PROCESSED / "elliptic.parquet"
EDGE_CACHE = C.DATA_PROCESSED / "edges.parquet"


# --------------------------------------------------------------------------- load
def _read_kaggle(raw):
    f = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    f.columns = ["txId", "time_step"] + FEATURE_COLS
    c = pd.read_csv(raw / "elliptic_txs_classes.csv")
    c["class"] = c["class"].astype(str).map({"1": 1, "2": 2, "unknown": 3})
    e = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
    return f, c, e


def _read_ellipticpp(raw):
    f = pd.read_csv(raw / "txs_features.csv")
    f = f.iloc[:, : 2 + C.N_LOCAL + C.N_AGG]           # keep only the original 166 columns
    f.columns = ["txId", "time_step"] + FEATURE_COLS
    c = pd.read_csv(raw / "txs_classes.csv")
    e = pd.read_csv(raw / "txs_edgelist.csv")
    return f, c, e


def load_raw(raw_dir=C.DATA_RAW, use_cache=True):
    """Return (nodes, edges). nodes has txId, time_step, 165 features and `label`
    where label = 1 illicit, 0 licit, -1 unknown (excluded from supervised training)."""
    if use_cache and CACHE.exists() and EDGE_CACHE.exists():
        return pd.read_parquet(CACHE), pd.read_parquet(EDGE_CACHE)
    if (raw_dir / "elliptic_txs_features.csv").exists():
        f, c, e = _read_kaggle(raw_dir)
    elif (raw_dir / "txs_features.csv").exists():
        f, c, e = _read_ellipticpp(raw_dir)
    else:
        raise FileNotFoundError(f"No Elliptic CSVs found in {raw_dir}. See data/README.md.")
    e.columns = ["txId1", "txId2"]
    nodes = f.merge(c, on="txId", how="left", validate="one_to_one").copy()
    nodes["label"] = nodes["class"].map({1: C.ILLICIT, 2: C.LICIT, 3: -1}).astype("int8")
    nodes = nodes.drop(columns="class")
    nodes["time_step"] = nodes["time_step"].astype("int16")
    nodes[FEATURE_COLS] = nodes[FEATURE_COLS].astype("float32")
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(CACHE)
    e.to_parquet(EDGE_CACHE)
    return nodes, e


# ----------------------------------------------------------------------- validate
def validate(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Integrity checks from slide 5. Returns a table; raises if a hard check fails."""
    step = nodes.set_index("txId")["time_step"]
    known = edges["txId1"].isin(step.index) & edges["txId2"].isin(step.index)
    cross = (step.reindex(edges["txId1"]).values != step.reindex(edges["txId2"]).values).sum()
    z = nodes[FEATURE_COLS]
    z = (z - z.mean()) / z.std(ddof=0)
    rows = [
        ("Transaction nodes", len(nodes), 203_769),
        ("Directed edges", len(edges), 234_355),
        ("Feature columns used (LF + AF)", len(FEATURE_COLS), 165),
        ("Missing feature values", int(nodes[FEATURE_COLS].isna().sum().sum()), 0),
        ("Duplicate txId", int(nodes["txId"].duplicated().sum()), 0),
        ("Duplicate edges", int(edges.duplicated().sum()), 0),
        ("Self-loops", int((edges["txId1"] == edges["txId2"]).sum()), 0),
        ("Edges to unknown txId", int((~known).sum()), 0),
        ("Edges crossing time steps", int(cross), 0),
        ("Time steps", nodes["time_step"].nunique(), 49),
        ("Illicit (class 1)", int((nodes["label"] == C.ILLICIT).sum()), 4_545),
        ("Licit (class 2)", int((nodes["label"] == C.LICIT).sum()), 42_019),
        ("Unknown", int((nodes["label"] == -1).sum()), 157_205),
        ("Values with |z| > 3 (reported, not removed)", int((z.abs() > 3).sum().sum()), None),
    ]
    rep = pd.DataFrame(rows, columns=["check", "observed", "expected"])
    rep["pass"] = [(r[2] is None) or (r[1] == r[2]) for r in rows]
    rep["expected"] = [("" if r[2] is None else r[2]) for r in rows]
    hard = rep.iloc[:9]
    if not hard["pass"].all():
        raise ValueError("Integrity check failed:\n" + hard.to_string())
    return rep


# ---------------------------------------------------------------- prepared object
@dataclass
class EllipticData:
    txid: np.ndarray
    time_step: np.ndarray
    X_raw: np.ndarray               # (N, 165) unscaled
    y: np.ndarray                   # 1 / 0 / -1
    edge_index: np.ndarray          # (2, E) row indices, source -> target
    X: np.ndarray = field(default=None)   # scaled with training-period statistics
    scaler: StandardScaler = field(default=None)

    lf_idx = np.arange(0, C.N_LOCAL)
    af_idx = np.arange(C.N_LOCAL, C.N_LOCAL + C.N_AGG)

    def mask(self, steps, labelled=True):
        m = np.isin(self.time_step, list(steps))
        return m & (self.y >= 0) if labelled else m

    def fit_scaler(self, fit_steps=C.TRAIN_STEPS):
        """StandardScaler fitted on the training period ONLY (all nodes in those steps,
        labelled or not, since features are observable at prediction time), then applied
        to every node. Never fitted on the test period."""
        fit_rows = self.mask(fit_steps, labelled=False)
        self.scaler = StandardScaler().fit(self.X_raw[fit_rows])
        self.X = self.scaler.transform(self.X_raw).astype(np.float32)
        return self

    def block(self, name):
        return {"LF": self.lf_idx, "ALL": np.r_[self.lf_idx, self.af_idx], "AF": self.af_idx}[name]


def prepare(raw_dir=C.DATA_RAW, fit_steps=C.TRAIN_STEPS) -> EllipticData:
    nodes, edges = load_raw(raw_dir)
    nodes = nodes.sort_values("txId").reset_index(drop=True)
    pos = pd.Series(np.arange(len(nodes)), index=nodes["txId"].values)
    ei = np.vstack([pos.loc[edges["txId1"]].values, pos.loc[edges["txId2"]].values])
    d = EllipticData(
        txid=nodes["txId"].values,
        time_step=nodes["time_step"].values.astype(int),
        X_raw=nodes[FEATURE_COLS].values.astype(np.float32),
        y=nodes["label"].values.astype(int),
        edge_index=ei.astype(np.int64),
    )
    return d.fit_scaler(fit_steps)


def split_summary(d: EllipticData) -> pd.DataFrame:
    rows = []
    for name, steps in [("train (1-34)", C.TRAIN_STEPS), ("test (35-49)", C.TEST_STEPS),
                        ("  tune-train (1-28)", C.TUNE_TRAIN_STEPS), ("  tune-val (29-34)", C.TUNE_VAL_STEPS)]:
        m = d.mask(steps)
        rows.append((name, int(m.sum()), int((d.y[m] == 1).sum()), int((d.y[m] == 0).sum()),
                     round(float((d.y[m] == 1).mean()) * 100, 2)))
    return pd.DataFrame(rows, columns=["split", "labelled", "illicit", "licit", "illicit_%"])
