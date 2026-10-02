# Central settings: paths, snapshot weeks for the time-based split, feature lists. Override the root with RETAINIQ_ROOT.
import os
from pathlib import Path

# Overridable so the same code runs in Colab, Docker (/app) and CI
ROOT = Path(os.environ.get("RETAINIQ_ROOT", "/content/retainiq"))
DB_PATH = str(ROOT / "retainiq.db")
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_STUDENTS = 6000
N_WEEKS = 20            # weeks of history in the DB
LABEL_WINDOW = 2        # label = no logins in the next 2 weeks after the snapshot

# Rolling-origin (time-based) split. Rule: a split's label window must END before the next split's snapshot.
TRAIN_SNAPSHOTS = [8, 9, 10]  # labels use weeks <= 12
VALID_SNAPSHOTS = [12]        # labels use weeks 13-14
TEST_SNAPSHOTS = [14]         # labels use weeks 15-16
SERVING_SNAPSHOT = 18     # "today": latest week we can score (no label yet)

NUMERIC_FEATURES = [
    "age", "logins_last4w", "logins_last2w", "logins_prev2w", "logins_last_week",
    "logins_early_avg", "login_trend", "drop_vs_baseline", "quiz_score_mean_4w",
    "assignments_4w", "video_minutes_4w", "forum_posts_4w", "weeks_since_active",
]
CATEGORICAL_FEATURES = ["plan", "signup_channel"]
TARGET = "churned"
TOP_K_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
