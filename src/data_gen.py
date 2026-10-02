# Synthetic EdTech data: students + weekly activity. A hidden 'engagement' level drives each student's weekly churn hazard.
import sqlite3
import numpy as np
import pandas as pd
from config import DB_PATH, N_STUDENTS, N_WEEKS, SEED


# Build the students and weekly_activity tables and write them to SQLite.
def generate(db_path=DB_PATH, n_students=N_STUDENTS, seed=SEED):
    rng = np.random.default_rng(seed)

    students = pd.DataFrame({
        "student_id": np.arange(1, n_students + 1),
        "plan": rng.choice(["free", "basic", "premium"], n_students, p=[0.5, 0.35, 0.15]),
        "signup_channel": rng.choice(["organic", "ads", "referral", "campus"], n_students,
                                     p=[0.35, 0.30, 0.15, 0.20]),
        "age": rng.integers(16, 45, n_students),
    })

    # Latent engagement (hidden from the model) sets each student's weekly churn hazard
    engagement = rng.beta(2.5, 2.5, n_students)
    plan_boost = students["plan"].map({"free": -0.10, "basic": 0.0, "premium": 0.10}).to_numpy()
    engagement = np.clip(engagement + plan_boost, 0.02, 0.98)
    hazard = 0.02 + 0.22 / (1 + np.exp(8 * (engagement - 0.30)))   # weekly drop probability

    # First week with zero activity (geometric waiting time; >= week 4)
    drop_week = rng.geometric(hazard) + 3
    # Some students fade gradually before dropping, others vanish suddenly
    fade_len = rng.integers(0, 4, n_students)

    rows = []
    for w in range(1, N_WEEKS + 1):
        weeks_to_drop = drop_week - w
        fade = np.where(weeks_to_drop <= fade_len,
                        np.clip(0.25 + 0.25 * (weeks_to_drop - 1), 0.05, 1.0), 1.0)
        active = (w < drop_week).astype(float)
        lam = (engagement * 6 + 0.5) * fade * active

        logins = rng.poisson(lam)
        quiz_attempts = rng.poisson(lam * 0.6)
        quiz_score = np.clip(rng.normal(40 + 45 * engagement, 12), 0, 100)
        quiz_score = np.where(quiz_attempts > 0, quiz_score, np.nan)
        assignments = rng.binomial(2, np.clip(engagement * fade * active, 0, 1))
        video_minutes = rng.gamma(2.0, 10 * (lam + 0.1))
        forum_posts = rng.poisson(lam * 0.15)

        rows.append(pd.DataFrame({
            "student_id": students["student_id"], "week": w, "logins": logins,
            "video_minutes": video_minutes.round(1), "quiz_attempts": quiz_attempts,
            "avg_quiz_score": quiz_score, "assignments_submitted": assignments,
            "forum_posts": forum_posts,
        }))
    activity = pd.concat(rows, ignore_index=True)

    con = sqlite3.connect(db_path)
    students.to_sql("students", con, if_exists="replace", index=False)
    activity.to_sql("weekly_activity", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_act ON weekly_activity(student_id, week)")
    con.commit()
    con.close()
    print(f"students={len(students)} activity_rows={len(activity)}")


if __name__ == "__main__":
    generate()
