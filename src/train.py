# Training pipeline: time-based split, LogReg baseline vs XGBoost, MLflow logging, model + drift-reference export.
import json
import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (ARTIFACT_DIR, CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED, TARGET,
                    TEST_SNAPSHOTS, TOP_K_FRACTIONS, TRAIN_SNAPSHOTS, VALID_SNAPSHOTS)
from features import build_dataset

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Median imputation (+ missing flags), optional scaling, one-hot for categories.
def make_preprocessor(scale: bool):
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


# Capacity-aware metrics: if the team contacts the top k%, what are precision, recall and lift?
def topk_metrics(y_true, scores, fracs=TOP_K_FRACTIONS):
    # Capacity-aware view: if the team contacts the top-k% riskiest students...
    y_true = np.asarray(y_true)
    order = np.argsort(-scores)
    base = y_true.mean()
    out = {}
    for f in fracs:
        k = max(1, int(len(scores) * f))
        hit = y_true[order[:k]].sum()
        out[f"precision_at_{int(f*100)}"] = float(hit / k)
        out[f"recall_at_{int(f*100)}"] = float(hit / y_true.sum())
        out[f"lift_at_{int(f*100)}"] = float((hit / k) / base)
    return out


# Score a dataset: ROC-AUC, PR-AUC, base rate and top-k metrics.
def evaluate(pipe, df):
    scores = pipe.predict_proba(df[FEATURES])[:, 1]
    y = df[TARGET].to_numpy()
    m = {"roc_auc": float(roc_auc_score(y, scores)),
         "pr_auc": float(average_precision_score(y, scores)),
         "base_rate": float(y.mean())}
    m.update(topk_metrics(y, scores))
    return m


# Command-line entry point.
def main():
    train = build_dataset(TRAIN_SNAPSHOTS)
    valid = build_dataset(VALID_SNAPSHOTS)
    test = build_dataset(TEST_SNAPSHOTS)
    print("rows  train/valid/test:", len(train), len(valid), len(test))
    print("churn train/valid/test:", round(train[TARGET].mean(), 3),
          round(valid[TARGET].mean(), 3), round(test[TARGET].mean(), 3))

    models = {
        "logreg_baseline": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "xgboost": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=SEED)),
        ]),
    }

    mlflow.set_tracking_uri(f"sqlite:///{ARTIFACT_DIR}/mlflow.db")
    mlflow.set_experiment("retainiq-v2")

    results, best_name, best_val = {}, None, -1
    for name, pipe in models.items():
        with mlflow.start_run(run_name=name):
            pipe.fit(train[FEATURES], train[TARGET])
            val_m = evaluate(pipe, valid)
            test_m = evaluate(pipe, test)
            mlflow.log_param("model", name)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
            results[name] = {"valid": val_m, "test": test_m}
            print(f"\n{name}")
            print("  valid pr_auc=%.3f  test pr_auc=%.3f  test roc_auc=%.3f" %
                  (val_m["pr_auc"], test_m["pr_auc"], test_m["roc_auc"]))
            for f in TOP_K_FRACTIONS:
                p = int(f * 100)
                print("  test top-%2d%%: precision=%.3f recall=%.3f lift=%.2f" %
                      (p, test_m[f"precision_at_{p}"], test_m[f"recall_at_{p}"], test_m[f"lift_at_{p}"]))
            # Model selection uses VALID only; test is reported, never used to choose
            if val_m["pr_auc"] > best_val:
                best_name, best_val = name, val_m["pr_auc"]

    print("\nSelected on validation PR-AUC:", best_name)
    joblib.dump(models[best_name], ARTIFACT_DIR / "model.joblib")
    meta = {"model": best_name, "features": FEATURES, "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES, "metrics": results[best_name]}
    (ARTIFACT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # Reference distribution for drift monitoring (training features only)
    train[NUMERIC_FEATURES].to_csv(ARTIFACT_DIR / "train_reference.csv", index=False)


if __name__ == "__main__":
    main()
