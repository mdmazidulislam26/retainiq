# Data-drift monitoring with PSI (Population Stability Index) against the training data.
import numpy as np
import pandas as pd
from config import ARTIFACT_DIR, NUMERIC_FEATURES, SERVING_SNAPSHOT
from features import build_feature_table


# Population Stability Index of one feature. Bin edges come from the training (expected) quantiles.
def psi(expected, actual, bins=10, eps=1e-4):
    # Population Stability Index. Bin edges come from the TRAINING (expected) quantiles.
    expected = pd.Series(expected).dropna().to_numpy()
    actual = pd.Series(actual).dropna().to_numpy()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # constant-ish feature: nothing to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


# Rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 alert.
def status(v):
    # Common rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 investigate/retrain
    return "stable" if v < 0.1 else ("watch" if v < 0.25 else "ALERT")


# PSI for every numeric feature, worst first.
def drift_report(current_df, reference_path=ARTIFACT_DIR / "train_reference.csv"):
    ref = pd.read_csv(reference_path)
    rows = []
    for col in NUMERIC_FEATURES:
        v = psi(ref[col], current_df[col])
        rows.append({"feature": col, "psi": round(v, 4), "status": status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    cur = build_feature_table(snap=SERVING_SNAPSHOT)
    print(drift_report(cur).to_string(index=False))
