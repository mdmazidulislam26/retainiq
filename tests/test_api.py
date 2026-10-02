# API contract tests via FastAPI TestClient.
from fastapi.testclient import TestClient
from api import app
from config import SERVING_SNAPSHOT
from features import build_feature_table

client = TestClient(app)


# Build a valid request body from real feature rows.
def sample_payload(n=3):
    df = build_feature_table(snap=SERVING_SNAPSHOT).head(n)
    # JSON cannot carry NaN -> send null (the pipeline imputes it)
    records = df.drop(columns=["snapshot"]).astype(object).where(df.notna(), None).to_dict("records")
    return {"students": records}


# /health responds with status ok.
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


# /predict returns probabilities between 0 and 1.
def test_predict_returns_valid_probabilities():
    r = client.post("/predict", json=sample_payload(3))
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3
    assert all(0.0 <= p["churn_risk"] <= 1.0 for p in preds)


# /explain returns three reasons per student.
def test_explain_returns_top_reasons():
    r = client.post("/explain", json=sample_payload(2))
    assert r.status_code == 200
    first = r.json()["explanations"][0]
    assert len(first["top_reasons"]) == 3


# Missing fields are rejected with HTTP 422.
def test_rejects_bad_payload():
    r = client.post("/predict", json={"students": [{"student_id": 1}]})
    assert r.status_code == 422
