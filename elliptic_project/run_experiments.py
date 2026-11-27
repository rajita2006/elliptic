"""Run the whole pipeline end to end from the command line.

    python run_experiments.py            # full protocol: tuning + 4 arms x 2 models x 5 seeds + ablation
    python run_experiments.py --quick    # smoke test: 1 seed, 1 config per model, 20 GNN epochs
    python run_experiments.py --analyse  # skip training, rebuild tables/figures from saved predictions
"""
import argparse

import torch

from src import analysis, config as C, data as D, elliptical, experiment, models as M

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--analyse", action="store_true")
args = ap.parse_args()
torch.set_num_threads(2)

d = D.prepare()
C.TABLES.mkdir(parents=True, exist_ok=True)
nodes, edges = D.load_raw()
D.validate(nodes, edges).to_csv(C.TABLES / "integrity_checks.csv", index=False)
D.split_summary(d).to_csv(C.TABLES / "split_summary.csv", index=False)

ell = elliptical.run(d)
ell["gof"].to_csv(C.TABLES / "elliptical_goodness_of_fit.csv", index=False)
ell["detector"].to_csv(C.TABLES / "elliptical_as_detector.csv", index=False)

if not args.analyse:
    seeds = C.SEEDS
    if args.quick:
        seeds = [0]
        for k in M.TREE_GRIDS: M.TREE_GRIDS[k] = M.TREE_GRIDS[k][:1]
        for k in M.GNN_GRIDS: M.GNN_GRIDS[k] = M.GNN_GRIDS[k][:1]
        M.MAX_EPOCHS = 20
    experiment.run_all(d, seeds=seeds, do_ablation=True)

analysis.run(d)
