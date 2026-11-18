"""Central configuration shared by training scripts and the API (single source of truth)."""
import os

SEED = 42

# Candidate generation (values re-checked by scripts/02_build_features.py (miss rate per trip))
CAND_RADIUS_M = 50.0     # keep arcs whose exact distance to the GPS point is <= this
MAX_CANDIDATES = 12      # keep at most K nearest arcs per observation
MIN_CANDIDATES = 3       # if fewer found inside radius, fall back to nearest arcs regardless of radius
SAMPLE_STEP_M = 15.0     # arc densification step used by the KD-tree spatial index

# Trip-level label QC (training partition only): a trip whose derived reference labels sit far from the GPS
# track (reference route does not describe the recorded drive) is excluded from model fitting/CV.
QC_MAX_MEAN_LABEL_DIST = 15.0
QC_MAX_MISS = 0.05

# Heading / motion estimation
HEADING_HALF_WINDOW = 2  # observations either side used for motion heading
MIN_SPEED_FOR_HEADING = 1.0  # m/s

# Default HMM parameters (re-tuned by scripts/05_hmm_tune_eval.py -> artifacts/hmm_config.json)
HMM_DEFAULT = dict(beta=10.0, tau=1.0, unreachable_penalty=15.0, backward_penalty=3.0,
                   dist_slack_m=150.0, dist_factor=2.0, gap_scale=True)

FEATURES = [
    "d", "d_rank", "d_gap", "d_ratio", "n_cand",
    "frac", "along", "arc_len", "remain",
    "cos_dh", "abs_dh", "heading_valid", "speed",
    "d_m1", "d_p1", "d_m2", "d_p2", "d_win_mean", "d_win_gap", "d_win_rank",
    "prog", "prog_ratio", "has_rev", "deg_to_out", "deg_from_in",
    "dt_prev", "dt_next",
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")
RESULT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
