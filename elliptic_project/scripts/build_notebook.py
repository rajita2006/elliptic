"""Generates notebooks/Elliptic_Illicit_Detection.ipynb (then executed with nbconvert)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CONCLUSIONS = (ROOT / "scripts" / "conclusions.md").read_text() if (ROOT / "scripts" / "conclusions.md").exists() else "_Run the notebook to fill in._"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Illicit Bitcoin Transaction Detection on the Elliptic Dataset
**BCSE209L — Machine Learning · VIT Vellore, SCOPE · B.Tech CSE (Data Science)**
Team: ADITHIYA SR (23BDS0045) · SHRIJANKIRITH T (23BDS0034) · RAJITA SK (23BDS0010)

This notebook runs the full system described in the project demonstration deck:
dataset → preprocessing → feature-block separation → tabular + graph representation →
four controlled detection arms → illicit/licit classification → per-time-step evaluation.

Section numbers follow the deck. Everything runs on a laptop CPU.
""")
code("""
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, torch
from IPython.display import Image, display
pd.set_option("display.max_columns", 30); pd.set_option("display.width", 220)
torch.set_num_threads(2)

from src import config as C, data as D, elliptical, models as M, experiment, analysis, trace

# Training all arms x 5 seeds takes ~1-2 h on a laptop CPU. The submitted results are
# saved in results/predictions.parquet; set RUN_TRAINING = True to regenerate them.
RUN_TRAINING = False
""")

md("""
## 1. Research objective
Classify Bitcoin transactions as illicit or licit under a protocol that (i) separates the local
transaction features (LF) from the neighbourhood-aggregated features (AF), (ii) compares tree
ensembles against graph neural networks with that confound removed, and (iii) reports
minority-class performance **per time step**, before and after the dark-market shutdown at t = 43.

> **Feature-count note.** The published "94 local features" include the time step. Slide 4 states that
> time step is used to build the split and is not a predictor, so this implementation uses
> **LF = 93 columns, AF = 72 columns, 165 predictors in total.**
""")

md("## 2–4. Input data and integrity validation")
code("""
nodes, edges = D.load_raw()
checks = D.validate(nodes, edges)   # raises if any hard check fails
checks
""")
code("""
import matplotlib.pyplot as plt
lab = nodes[nodes.label >= 0]
cnt = lab.groupby(["time_step", "label"]).size().unstack(fill_value=0).rename(columns={0: "licit", 1: "illicit"})
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.bar(cnt.index, cnt.licit, color="#4C78A8", label="licit"); ax.bar(cnt.index, cnt.illicit, bottom=cnt.licit, color="#E45756", label="illicit")
ax.axvline(34.5, color="k", ls="--"); ax.text(34.7, ax.get_ylim()[1]*.9, "test →")
ax.axvline(42.5, color="crimson", ls=":"); ax.text(42.7, ax.get_ylim()[1]*.75, "t=43 shutdown", color="crimson")
ax.set(xlabel="time step", ylabel="labelled transactions", title="Labelled transactions per time step"); ax.legend(); plt.show()
print(f"Illicit share of ALL nodes: {(nodes.label==1).mean():.1%}  |  of labelled nodes: {(lab.label==1).mean():.1%}  |  unlabelled: {(nodes.label==-1).mean():.1%}")
""")

md("""
## 5. Preprocessing
* Missing values / duplicates: validated above (none).
* Target encoding: class 1 → illicit (1), class 2 → licit (0), unknown → excluded from supervised training.
* **Scaling**: `StandardScaler` fitted on time steps 1–34 only, then applied to the test period.
* **Outliers**: counted and reported, never removed (they may be the illicit cases).
* **Split**: frozen temporal split, train t1–34, test t35–49. Hyper-parameters are tuned on a
  validation slice inside the training period (fit t1–28, validate t29–34).
* **Imbalance**: one illicit-class weight (= #licit / #illicit in training) used in every arm's loss.
""")
code("""
d = D.prepare()
display(D.split_summary(d))
w = M.illicit_weight(d.y[d.mask(C.TRAIN_STEPS)])
print(f"Illicit class weight shared by all arms: {w:.3f}")
print("Scaler fitted on", int(d.mask(C.TRAIN_STEPS, labelled=False).sum()), "training-period nodes; test-period mean of LF_1 after scaling =",
      round(float(d.X[d.mask(C.TEST_STEPS, labelled=False), 0].mean()), 3), "(not 0, as expected when no test statistics leak in)")
""")

md("""
## 6. Data representation and the elliptical-distribution question
"Elliptic" is the company that released the data, not a statement about elliptical distributions.
We **test** the assumption: fit mean and covariance on licit training transactions (LF block),
compute squared Mahalanobis distances, and compare them with the chi-square distribution they would
follow if licit points formed one ellipsoid. Then we use the distance as an unsupervised detector.
""")
code("""
ell = elliptical.run(d)
display(ell["gof"]); display(ell["detector"])
""")
code("""
for f in ["elliptical_pca_ellipse.png", "elliptical_chi2_qq.png", "elliptical_distance_hist.png"]:
    display(Image(filename=str(C.FIGURES / f), width=650))
""")

md("""
## 7. Proposed detection method — four controlled arms
| Arm | Model family | Input | Question it answers |
|---|---|---|---|
| A1 | XGBoost / Random Forest | LF (93) | non-graph baseline |
| A2 | XGBoost / Random Forest | LF + AF (165) | value of precomputed 1-hop aggregates (A2 vs A1) |
| A3 | GraphSAGE / GCN | LF (93) + edges | learned message passing without aggregates (A3 vs A2) |
| A4 | GraphSAGE / GCN | LF + AF + edges | are aggregates and message passing redundant? |

Held constant: split, class weight, 5 seeds (0–4), threshold 0.5, illicit-class metrics.
GNNs: 2 graph-conv layers + linear head, ReLU, dropout 0.3, Adam (weight decay 5e-4), full-batch,
edges used in both directions. Each period is its own sub-graph because no edge crosses a time step.
""")
code("""
if RUN_TRAINING:
    experiment.run_all(d, seeds=C.SEEDS, do_ablation=True)
cfg = json.load(open(C.RESULTS / "run_config.json"))
display(pd.DataFrame(cfg["best_params"].items(), columns=["arm/model", "selected hyper-parameters (validation t29-34)"]))
tuning = pd.read_csv(C.TABLES / "tuning_results.csv")
display(tuning.sort_values(["arm", "model", "val_PR_AUC"], ascending=[True, True, False]).groupby(["arm", "model"]).head(2))
""")

md("## 9–11. Results: classification and per-time-step evaluation")
code("""
summary = analysis.run(d)   # rebuilds every table and figure from the saved predictions
main = pd.read_csv(C.TABLES / "main_results.csv")
main[main.sampling == "weights"].drop(columns="sampling")
""")
md("### Per-time-step illicit F1 (the central figure)")
code("""display(Image(filename=str(C.FIGURES / "drift_curve_illicit_f1.png")))
pd.read_csv(C.TABLES / "per_step_f1_wide.csv", header=[0, 1], index_col=0)""")
md("### Pre-drift vs post-drift")
code("""display(Image(filename=str(C.FIGURES / "pre_vs_post_f1.png"), width=850))
main[main.sampling == "weights"][["arm", "model", "F1 pre t35-42", "F1 post t43-49", "post - pre"]]""")
md("### Precision–recall curves")
code("""display(Image(filename=str(C.FIGURES / "pr_curves.png"), width=560))""")
md("### Confusion matrices per time step (representative model per arm, mean over seeds)")
code("""
rep = json.load(open(C.RESULTS / "analysis_meta.json"))["representative"]
cm = pd.read_csv(C.TABLES / "per_step_f1_and_confusion.csv")
cm = cm[(cm.sampling == "weights") & cm.apply(lambda r: rep[r.arm] == r.model, axis=1)]
cm.pivot_table(index="time_step", columns="arm", values=["tp", "fp", "fn"]).round(1)
""")
md("### Feature-block attribution before vs after the change point")
code("""display(Image(filename=str(C.FIGURES / "feature_block_attribution.png"), width=800))
display(pd.read_csv(C.TABLES / "feature_block_attribution.csv"))
pd.read_csv(C.TABLES / "top_shap_features_pre_post.csv").pivot(index="rank", columns="window", values="feature")""")
md("### Sampling ablation: class weights vs NCL-SMOTE (XGBoost, A1 and A2)")
code("""main[main.model == "XGBoost"][["arm", "sampling", "illicit_F1 (pooled)", "PR_AUC", "F1 pre t35-42", "F1 post t43-49"]]""")

md("## 8. System workflow demonstration — one real test transaction traced end to end")
md("**(a) An illicit transaction the system catches**")
code("""
pd.set_option("display.max_colwidth", None)
tx = trace.pick_example(d, step=38, caught=True)
trace.trace(d, tx, rep)
""")
md("**(b) An illicit transaction it misses** — useful for error analysis")
code("""
tx = trace.pick_example(d, step=38, caught=False)
trace.trace(d, tx, rep)
""")

md("## Ranked alert list (expected output, slide 9)")
code("""
meta = json.load(open(C.RESULTS / "analysis_meta.json"))
arm, model = meta["alert_model"].split("/")
alerts = pd.read_csv(C.RESULTS / f"alert_list_{arm}_{model}.csv")
print(f"Alert model (best validation PR-AUC): {meta['alert_model']}")
for k in [50, 100, 250, 500, 1000]:
    print(f"  top {k:>4}: precision {alerts.cum_precision.iloc[k-1]:.3f}, recall {alerts.cum_recall.iloc[k-1]:.3f}")
alerts.head(15)
""")

md("## Conclusions\n" + CONCLUSIONS)

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebooks" / "Elliptic_Illicit_Detection.ipynb")
print("notebook written")
