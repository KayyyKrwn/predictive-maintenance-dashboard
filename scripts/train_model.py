"""Train failure-prediction model on AI4I 2020 + derive maintenance rules.

Outputs (in ../models/):
  - pipeline.pkl        : sklearn Pipeline (preprocess + RandomForest)
  - metrics.json        : evaluation metrics on holdout set
  - eda_summary.json    : key EDA stats for the dashboard
  - maintenance_rules.json : data-driven preventive-maintenance thresholds
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "ai4i2020.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

# ---------------- Load & clean ----------------
df = pd.read_csv(DATA, encoding="utf-8-sig")
df.columns = (df.columns.str.strip()
              .str.replace(r" \[K\]$", "_K", regex=True)
              .str.replace(r" \[rpm\]$", "_rpm", regex=True)
              .str.replace(r" \[Nm\]$", "_Nm", regex=True)
              .str.replace(r" \[min\]$", "_min", regex=True)
              .str.replace(" ", "_"))
# -> UDI, Product_ID, Type, Air_temperature_K, Process_temperature_K,
#    Rotational_speed_rpm, Torque_Nm, Tool_wear_min, Machine_failure,
#    TWF, HDF, PWF, OSF, RNF

# ---------------- Feature engineering ----------------
df["Temp_diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
df["Power_W"] = 2 * np.pi * df["Rotational_speed_rpm"] * df["Torque_Nm"] / 60.0

FEATURES = ["Type", "Air_temperature_K", "Process_temperature_K",
            "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
            "Temp_diff_K", "Power_W"]
TARGET = "Machine_failure"
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
MODE_NAMES = {"TWF": "Tool Wear Failure", "HDF": "Heat Dissipation Failure",
              "PWF": "Power Failure", "OSF": "Overstrain Failure",
              "RNF": "Random Failure"}

# ---------------- EDA summary ----------------
n = len(df)
n_fail = int(df[TARGET].sum())
mode_counts = {m: int(df[m].sum()) for m in MODES}
fail_by_type = df.groupby("Type")[TARGET].agg(["sum", "count"])
fail_by_type["rate"] = fail_by_type["sum"] / fail_by_type["count"]

eda = {
    "n_rows": n,
    "n_failures": n_fail,
    "failure_rate": round(n_fail / n, 4),
    "mode_counts": mode_counts,
    "mode_names": MODE_NAMES,
    "failure_rate_by_type": {t: round(r, 4) for t, r in fail_by_type["rate"].items()},
    "failures_by_type": {t: int(c) for t, c in fail_by_type["sum"].items()},
    "feature_ranges": {c: [round(float(df[c].min()), 2), round(float(df[c].max()), 2)]
                       for c in FEATURES if c != "Type"},
    "feature_medians": {c: round(float(df[c].median()), 2)
                        for c in FEATURES if c != "Type"},
}
(MODELS / "eda_summary.json").write_text(json.dumps(eda, indent=2))
print(f"rows={n} failures={n_fail} ({n_fail/n:.2%})")
print("modes:", mode_counts)
print(fail_by_type)

# ---------------- Train / test ----------------
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

num_feats = [c for c in FEATURES if c != "Type"]
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
    ("num", "passthrough", num_feats),
])
clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                             random_state=42, n_jobs=-1)
pipe = Pipeline([("pre", pre), ("clf", clf)])
pipe.fit(X_train, y_train)

proba = pipe.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
metrics = {
    "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "avg_precision": round(float(average_precision_score(y_test, proba)), 4),
    "report_05": classification_report(y_test, pred, output_dict=True,
                                       zero_division=0),
    "confusion_05": confusion_matrix(y_test, pred).tolist(),
    "n_test": len(y_test),
}
(MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("ROC-AUC:", metrics["roc_auc"], "| AP:", metrics["avg_precision"])
print(classification_report(y_test, pred, zero_division=0))

joblib.dump(pipe, MODELS / "pipeline.pkl")

# Global feature importance (for dashboard)
ohe_names = list(pipe.named_steps["pre"].named_transformers_["cat"]
                 .get_feature_names_out(["Type"]))
feat_names = ohe_names + num_feats
importances = pipe.named_steps["clf"].feature_importances_
fi = sorted(zip(feat_names, importances), key=lambda t: -t[1])
(MODELS / "feature_importance.json").write_text(
    json.dumps([{"feature": f, "importance": round(float(i), 4)} for f, i in fi],
               indent=2))
print("top features:", fi[:5])

# ---------------- Maintenance rules (data-driven) ----------------
# Tool-wear ranges where TWF actually happened, per product Type
twf = df[df["TWF"] == 1]
tool_wear_limits = {}
for t in ["L", "M", "H"]:
    vals = twf.loc[twf["Type"] == t, "Tool_wear_min"]
    if len(vals):
        # recommend replacement at 90% of the lowest observed TWF wear
        tool_wear_limits[t] = {
            "min_twf_wear": round(float(vals.min()), 1),
            "replace_before_min": round(float(vals.min() * 0.9), 1),
        }

hdf = df[df["HDF"] == 1]
osf = df[df["OSF"] == 1]
pwf = df[df["PWF"] == 1]
rules = {
    "tool_wear_limits": tool_wear_limits,
    "hdf_temp_diff_threshold": round(float(hdf["Temp_diff_K"].quantile(0.1)), 2),
    "osf_strain_proxy_threshold": round(float((osf["Torque_Nm"] * osf["Tool_wear_min"]).quantile(0.1)), 1),
    "pwf_power_low": round(float(pwf["Power_W"].quantile(0.9)), 1),
    "pwf_power_high": round(float(pwf["Power_W"].quantile(0.1)), 1),
    "notes": {
        "TWF": "Ganti pahat sebelum mencapai 90% dari keausan minimum yang pernah menyebabkan TWF.",
        "HDF": "Periksa sistem pendingin bila selisih suhu proses-udara di atas threshold.",
        "PWF": "Periksa pasokan daya bila power di luar rentang normal.",
        "OSF": "Kurangi beban bila kombinasi torsi x keausan pahat tinggi.",
    },
}
(MODELS / "maintenance_rules.json").write_text(json.dumps(rules, indent=2))
print("rules:", json.dumps(rules, indent=2))
print("DONE")
