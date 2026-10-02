# Streamlit dashboard: weekly contact worklist sized to team capacity, plus the drift table.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st

from api import CLF, EXPLAINER, FEATURE_NAMES, FEATURES, MODEL, PREP
from config import SERVING_SNAPSHOT
from features import build_feature_table
from monitor import drift_report

st.set_page_config(page_title="RetainIQ", layout="wide")
st.title("RetainIQ - Weekly Retention Worklist")


@st.cache_data
def load_scored(snap: int):
    df = build_feature_table(snap=snap)
    df["churn_risk"] = MODEL.predict_proba(df[FEATURES])[:, 1]
    return df.sort_values("churn_risk", ascending=False).reset_index(drop=True)


# Main SHAP driver per student.
def top_reason(rows: pd.DataFrame) -> list:
    # Top SHAP driver per student
    X = PREP.transform(rows[FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    sv = EXPLAINER.shap_values(X)
    return [FEATURE_NAMES[int(np.argmax(np.abs(sv[i])))] for i in range(len(rows))]


scored = load_scored(SERVING_SNAPSHOT)
capacity = st.sidebar.slider("Team capacity (% of active students)", 1, 30, 10)
k = max(1, int(len(scored) * capacity / 100))
worklist = scored.head(k).copy()

c1, c2, c3 = st.columns(3)
c1.metric("Active students scored", f"{len(scored):,}")
c2.metric("Contact list size", f"{k:,}")
c3.metric("Avg risk in list", f"{worklist['churn_risk'].mean():.2f}")

st.subheader("Contact these students first")
if EXPLAINER is not None:
    worklist["main_driver"] = top_reason(worklist)
cols = ["student_id", "churn_risk", "plan", "logins_last2w", "weeks_since_active"] + \
       (["main_driver"] if EXPLAINER is not None else [])
st.dataframe(worklist[cols].round(3), use_container_width=True, hide_index=True)
st.download_button("Download worklist (CSV)", worklist[cols].to_csv(index=False), "worklist.csv")

st.subheader("Data drift vs training data (PSI)")
report = drift_report(scored)
st.dataframe(report, use_container_width=True, hide_index=True)
if (report["status"] == "ALERT").any():
    st.warning("At least one feature has drifted. Check before trusting this week's ranking.")
