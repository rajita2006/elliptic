Numbers are mean ± sd over 5 seeds, test period t35–49, threshold 0.5 (see `results/tables/main_results.csv`).

1. **The elliptical assumption fails, and distance from "normal" points the wrong way.** Squared Mahalanobis
   distances of licit training points do not follow chi-square (KS = 0.75, p ≈ 0; Mardia kurtosis ≈ 27× its
   expected value). Used as an unsupervised detector, d² scores *below* random (PR-AUC 0.038 vs 0.065 base rate)
   because illicit transactions sit in a dense region close to the licit centroid. This reproduces
   Lorenz et al. (2020) and is the reason a supervised method is needed.
2. **Aggregated features are worth about +0.05–0.07 F1 to tree models (A2 vs A1).** Random Forest rises from
   0.743 to 0.808 pooled illicit F1, mostly through precision (0.75 → 0.92). A2 Random Forest reproduces
   Weber et al.'s published 0.788 (RF on all features).
3. **Learned message passing does not beat precomputed aggregates (A3 vs A2).** The best GNN on local features
   (GraphSAGE, 0.557) is well below the tree ensemble given the 1-hop aggregates (0.79–0.81), and below even the
   no-graph tree baseline A1 (0.743). GNNs also vary more across seeds (sd 0.03–0.06 vs ≤ 0.007).
4. **For GNNs, aggregates and message passing overlap (A4 vs A3).** Adding AF raises thresholded F1
   (GraphSAGE 0.557 → 0.615, GCN 0.381 → 0.591), mainly by raising precision. But GraphSAGE's PR-AUC falls
   (0.669 → 0.588), so its overall ranking gets worse. The evidence is mixed: much of what the aggregates
   carry is already learned by message passing. Neither GNN reaches the tree arms. GraphSAGE beats GCN in both
   GNN arms.
5. **Every arm collapses after the dark-market shutdown.** Mean illicit F1 falls from 0.51–0.88 in t35–42 to
   0.02–0.05 in t43–49 for all eight arm/model combinations, consistent with Oliveira et al. (< 0.25 after t43).
   A single averaged F1 hides this completely: the pooled 0.81 of the best arm is driven by the pre-change window.
6. **The drop is not a switch of signal.** Attribution mass on the local vs aggregated block is nearly unchanged
   across the change point (A2 XGBoost: 64% / 36% before vs 63% / 37% after; A4 GraphSAGE about 50 / 50 in both),
   and the same features stay on top (LF_53, LF_59, AF_32). The models keep looking at the same evidence; after
   t43 that evidence no longer separates the new illicit activity.
7. **Sampling.** NCL-SMOTE instead of class weights gives a small, consistent gain for XGBoost
   (A2: 0.792 → 0.814 F1, PR-AUC 0.801 → 0.816) and does not change any of the conclusions above.

**Limitations.** Hyper-parameter grids are small (CPU budget); GNNs are full-batch 2-layer models with no temporal
component; results are for the labelled 23% of transactions only; the threshold is fixed at 0.5 rather than
tuned per deployment. Column names `LF_k` / `AF_k` are positional: `LF_k` is column k+2 of the Kaggle features file.
