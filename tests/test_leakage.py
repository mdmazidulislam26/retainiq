# Leakage and split-integrity tests (the most important tests in this project).
import shutil
import sqlite3
import pandas as pd
from config import DB_PATH
from features import build_feature_table, build_labels

SNAP = 8


# Delete all post-snapshot rows: features must not change (no future leakage).
def test_features_ignore_future_rows(tmp_path):
    # Copy the DB, DELETE every row after the snapshot: features must be identical
    tmp_db = str(tmp_path / "trimmed.db")
    shutil.copy(DB_PATH, tmp_db)
    con = sqlite3.connect(tmp_db)
    con.execute("DELETE FROM weekly_activity WHERE week > ?", (SNAP,))
    con.commit()
    con.close()
    full = build_feature_table(DB_PATH, SNAP).sort_values("student_id").reset_index(drop=True)
    trimmed = build_feature_table(tmp_db, SNAP).sort_values("student_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(full, trimmed)


# Labels are 0/1 and one per student.
def test_labels_binary_and_unique():
    labels = build_labels(snap=SNAP)
    assert set(labels["churned"].unique()) <= {0, 1}
    assert labels["student_id"].is_unique


# Feature table has exactly one row per student.
def test_one_row_per_student():
    assert build_feature_table(snap=SNAP)["student_id"].is_unique


# Each split's label window must end before the next split's snapshot.
def test_split_label_windows_do_not_overlap_next_snapshot():
    # Rolling-origin rule: each split's label window must end before the next split's snapshot
    from config import LABEL_WINDOW, TRAIN_SNAPSHOTS, VALID_SNAPSHOTS, TEST_SNAPSHOTS
    assert max(TRAIN_SNAPSHOTS) + LABEL_WINDOW <= min(VALID_SNAPSHOTS)
    assert max(VALID_SNAPSHOTS) + LABEL_WINDOW <= min(TEST_SNAPSHOTS)
