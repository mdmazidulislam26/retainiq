# FastAPI service: /health, /predict (risk scores), /explain (top SHAP reasons per student).
import json
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import ARTIFACT_DIR

app = FastAPI(title="RetainIQ API", version="2.0")

MODEL = joblib.load(ARTIFACT_DIR / "model.joblib")
META = json.loads((ARTIFACT_DIR / "model_meta.json").read_text())
FEATURES = META["features"]
PREP = MODEL.named_steps["prep"]
CLF = MODEL.named_steps["clf"]
IS_TREE = META["model"] == "xgboost"
EXPLAINER = shap.TreeExplainer(CLF) if IS_TREE else None
try:
    FEATURE_NAMES = list(PREP.get_feature_names_out())
except Exception:
    FEATURE_NAMES = None


# One student's features, as the API receives them (optional fields may be null).
class Student(BaseModel):
    student_id: int
    plan: str
    signup_channel: str
    age: float
    logins_last4w: float
    logins_last2w: float
    logins_prev2w: float
    logins_last_week: float
    logins_early_avg: Optional[float] = None
    login_trend: float
    drop_vs_baseline: Optional[float] = None
    quiz_score_mean_4w: Optional[float] = None
    assignments_4w: float
    video_minutes_4w: float
    forum_posts_4w: float
    weeks_since_active: Optional[float] = None


# A batch of students (1 to 5000 per request).
class Batch(BaseModel):
    students: List[Student] = Field(..., min_length=1, max_length=5000)


# Convert the request body to a DataFrame.
def to_frame(batch: Batch) -> pd.DataFrame:
    return pd.DataFrame([s.model_dump() for s in batch.students])


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model"], "n_features": len(FEATURES)}


@app.post("/predict")
def predict(batch: Batch):
    df = to_frame(batch)
    scores = MODEL.predict_proba(df[FEATURES])[:, 1]
    return {"predictions": [
        {"student_id": int(i), "churn_risk": round(float(s), 4)}
        for i, s in zip(df["student_id"], scores)
    ]}


@app.post("/explain")
def explain(batch: Batch, top_n: int = 3):
    if not IS_TREE:
        raise HTTPException(501, "SHAP explanations are only wired for the tree model")
    df = to_frame(batch)
    X = PREP.transform(df[FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    shap_vals = EXPLAINER.shap_values(X)
    scores = MODEL.predict_proba(df[FEATURES])[:, 1]
    out = []
    for r in range(len(df)):
        order = np.argsort(-np.abs(shap_vals[r]))[:top_n]
        reasons = [{"feature": FEATURE_NAMES[j],
                    "impact": round(float(shap_vals[r][j]), 4),
                    "direction": "raises risk" if shap_vals[r][j] > 0 else "lowers risk"}
                   for j in order]
        out.append({"student_id": int(df["student_id"].iloc[r]),
                    "churn_risk": round(float(scores[r]), 4),
                    "top_reasons": reasons})
    return {"explanations": out}
