# Data

The CSVs are not included (≈ 700 MB). Put ONE of the two releases in `data/raw/`:

**Option A — Kaggle (as cited on slide 10)**
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set → download, unzip so that you have

```
data/raw/elliptic_txs_features.csv
data/raw/elliptic_txs_classes.csv
data/raw/elliptic_txs_edgelist.csv
```

**Option B — Elliptic++ (the copy used for the submitted results)**
Same 203,769 transactions, 234,355 edges, 4,545 / 42,019 labels and the same 166 original columns
(plus 17 extra columns that the loader drops). Hosted on GitHub, no Kaggle login needed:

```
python scripts/download_data.py
```

`src/data.py` detects which release is present. The first load writes a parquet cache to
`data/processed/` so later loads take seconds. `run_experiments.py` and the notebook both run the
integrity checks (row/edge counts, no nulls, no duplicates, no cross-time-step edges) and stop if
they fail, so you can verify that either release gives the identical input.
