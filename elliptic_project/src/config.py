"""Central configuration. Everything that must be held constant across the four arms lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

# ---- Frozen temporal split (slide 5) ---------------------------------------
TRAIN_STEPS = range(1, 35)        # 1..34
TEST_STEPS = range(35, 50)        # 35..49
# Validation slice carved from the END of the training period, used only for
# hyper-parameter selection (slide 7). Final models are refit on 1..34.
TUNE_TRAIN_STEPS = range(1, 29)   # 1..28
TUNE_VAL_STEPS = range(29, 35)    # 29..34

# ---- Drift windows (slide 11) ----------------------------------------------
CHANGE_POINT = 43                 # dark-market shutdown
PRE_WINDOW = range(35, 43)        # 35..42
POST_WINDOW = range(43, 50)       # 43..49

# ---- Protocol ----------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
THRESHOLD = 0.50
ILLICIT, LICIT = 1, 0             # internal encoding of the target

# ---- Feature blocks ------------------------------------------------------------
# The published "94 local features" count includes the time step. Time step is
# used to build the split and is NOT a predictor (slide 4), so LF has 93 columns.
N_LOCAL = 93
N_AGG = 72

# ---- Published baselines (slide 11) -----------------------------------------
WEBER_RF_F1 = 0.788
