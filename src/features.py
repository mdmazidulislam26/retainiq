# SQL feature layer. Features use only weeks <= snapshot; labels use only the weeks after it (no leakage).
import sqlite3
import pandas as pd
from config import DB_PATH, LABEL_WINDOW

# Only rows with week <= :snap are visible -> no future leakage.
# All features are rolling windows relative to the snapshot, so they stay comparable
# across snapshots (an "all-time total" would silently grow with the snapshot week).
FEATURE_SQL = """
WITH base AS (
    SELECT * FROM weekly_activity WHERE week <= :snap
),
agg AS (
    SELECT
        student_id,
        SUM(CASE WHEN week >  :snap - 4 THEN logins ELSE 0 END)           AS logins_last4w,
        SUM(CASE WHEN week >  :snap - 2 THEN logins ELSE 0 END)           AS logins_last2w,
        SUM(CASE WHEN week >  :snap - 4 AND week <= :snap - 2
                 THEN logins ELSE 0 END)                                  AS logins_prev2w,
        SUM(CASE WHEN week =  :snap THEN logins ELSE 0 END)               AS logins_last_week,
        -- fixed 4-week baseline window: same meaning at every snapshot (an expanding window caused drift)
        AVG(CASE WHEN week >= :snap - 7 AND week <= :snap - 4
                 THEN logins END)                                         AS logins_early_avg,
        AVG(CASE WHEN week >  :snap - 4 THEN avg_quiz_score END)          AS quiz_score_mean_4w,
        SUM(CASE WHEN week >  :snap - 4 THEN assignments_submitted ELSE 0 END) AS assignments_4w,
        SUM(CASE WHEN week >  :snap - 4 THEN video_minutes ELSE 0 END)    AS video_minutes_4w,
        SUM(CASE WHEN week >  :snap - 4 THEN forum_posts ELSE 0 END)      AS forum_posts_4w,
        MAX(CASE WHEN logins > 0 THEN week END)                           AS last_active_week
    FROM base
    GROUP BY student_id
)
SELECT
    s.student_id, s.plan, s.signup_channel, s.age,
    a.logins_last4w, a.logins_last2w, a.logins_prev2w, a.logins_last_week,
    a.logins_early_avg, a.quiz_score_mean_4w, a.assignments_4w,
    a.video_minutes_4w, a.forum_posts_4w,
    :snap - a.last_active_week AS weeks_since_active
FROM students s
LEFT JOIN agg a USING (student_id)
"""

# Label comes ONLY from the future window (snap < week <= snap + window)
LABEL_SQL = """
SELECT student_id,
       CASE WHEN SUM(logins) = 0 THEN 1 ELSE 0 END AS churned
FROM weekly_activity
WHERE week > :snap AND week <= :snap + :win
GROUP BY student_id
"""


# Rolling-window features for one snapshot week; only students still active are scored.
def build_feature_table(db_path=DB_PATH, snap=8):
    con = sqlite3.connect(db_path)
    feats = pd.read_sql(FEATURE_SQL, con, params={"snap": snap})
    con.close()
    feats["login_trend"] = feats["logins_last2w"] / (feats["logins_prev2w"] + 1)
    # Recent weekly rate relative to the student's own early baseline
    feats["drop_vs_baseline"] = (feats["logins_last2w"] / 2) / (feats["logins_early_avg"] + 0.1)
    # Scoring only makes sense for students still active; already-gone ones are not "at risk"
    feats = feats[feats["logins_last2w"] > 0].reset_index(drop=True)
    feats["snapshot"] = snap
    return feats


# Label = 1 if the student has zero logins in the window right after the snapshot.
def build_labels(db_path=DB_PATH, snap=8, win=LABEL_WINDOW):
    con = sqlite3.connect(db_path)
    labels = pd.read_sql(LABEL_SQL, con, params={"snap": snap, "win": win})
    con.close()
    return labels


# Features + labels for one or more snapshots, stacked into one table.
def build_dataset(snapshots, db_path=DB_PATH):
    parts = []
    for s in snapshots:
        parts.append(build_feature_table(db_path, s).merge(build_labels(db_path, s), on="student_id"))
    return pd.concat(parts, ignore_index=True)
