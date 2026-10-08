"""Smart Manufacturing Analytics & Predictive Maintenance Dashboard.

Dataset : AI4I 2020 Predictive Maintenance (Kaggle)
Model   : RandomForest (class-balanced) -> probabilitas kegagalan mesin
Jalankan: .venv/bin/python -m streamlit run app.py
"""
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

matplotlib.use("Agg")
BASE = Path(__file__).resolve().parent

FEATURES = ["Type", "Air_temperature_K", "Process_temperature_K",
            "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
            "Temp_diff_K", "Power_W"]

LABEL_ID = {
    "Type_L": "Tipe produk: Light (L)", "Type_M": "Tipe produk: Medium (M)",
    "Type_H": "Tipe produk: Heavy (H)",
    "Air_temperature_K": "Suhu udara (K)",
    "Process_temperature_K": "Suhu proses (K)",
    "Rotational_speed_rpm": "Kecepatan putar (rpm)",
    "Torque_Nm": "Torsi (Nm)",
    "Tool_wear_min": "Keausan pahat (menit)",
    "Temp_diff_K": "Selisih suhu proses-udara (K)",
    "Power_W": "Daya (Watt)",
}
INPUT_LABEL_ID = {
    "Type": "Tipe produk",
    "Air_temperature_K": "Suhu udara (K)",
    "Process_temperature_K": "Suhu proses (K)",
    "Rotational_speed_rpm": "Kecepatan putar (rpm)",
    "Torque_Nm": "Torsi (Nm)",
    "Tool_wear_min": "Keausan pahat (menit)",
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip()
                  .str.replace(r" \[K\]$", "_K", regex=True)
                  .str.replace(r" \[rpm\]$", "_rpm", regex=True)
                  .str.replace(r" \[Nm\]$", "_Nm", regex=True)
                  .str.replace(r" \[min\]$", "_min", regex=True)
                  .str.replace(" ", "_"))
    return df


@st.cache_resource
def load_artifacts():
    pipe = joblib.load(BASE / "models" / "pipeline.pkl")
    eda = json.loads((BASE / "models" / "eda_summary.json").read_text())
    metrics = json.loads((BASE / "models" / "metrics.json").read_text())
    fi = json.loads((BASE / "models" / "feature_importance.json").read_text())
    rules = json.loads((BASE / "models" / "maintenance_rules.json").read_text())
    return pipe, eda, metrics, fi, rules


@st.cache_data
def load_data():
    df = clean_columns(pd.read_csv(BASE / "data" / "ai4i2020.csv",
                                   encoding="utf-8-sig"))
    df["Temp_diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    df["Power_W"] = 2 * np.pi * df["Rotational_speed_rpm"] * df["Torque_Nm"] / 60.0
    return df


@st.cache_resource
def get_explainer(_pipe):
    return shap.TreeExplainer(_pipe.named_steps["clf"])


def transformed_feature_names(pipe):
    ohe = list(pipe.named_steps["pre"].named_transformers_["cat"]
               .get_feature_names_out(["Type"]))
    num = [c for c in FEATURES if c != "Type"]
    return ohe + num


def predict_row(pipe, row: dict):
    X = pd.DataFrame([row])[FEATURES]
    return float(pipe.predict_proba(X)[0, 1])


def risk_band(p: float):
    if p >= 0.60:
        return "TINGGI", "#d62728"
    if p >= 0.25:
        return "SEDANG", "#ff7f0e"
    return "RENDAH", "#2ca02c"


def rule_warnings(row: dict, rules: dict):
    warns = []
    lim = rules["tool_wear_limits"].get(row["Type"], {})
    if lim and row["Tool_wear_min"] >= lim["replace_before_min"]:
        warns.append(f"Keausan pahat {row['Tool_wear_min']:.0f} mnt "
                     f"mendekati batas aman tipe {row['Type']} "
                     f"({lim['replace_before_min']:.0f} mnt) — risiko TWF.")
    if row["Temp_diff_K"] >= rules["hdf_temp_diff_threshold"]:
        warns.append("Selisih suhu tinggi — periksa sistem pendingin (risiko HDF).")
    if row["Power_W"] <= rules["pwf_power_low"] or row["Power_W"] >= rules["pwf_power_high"]:
        warns.append("Daya di luar rentang normal — periksa pasokan daya (risiko PWF).")
    if row["Torque_Nm"] * row["Tool_wear_min"] >= rules["osf_strain_proxy_threshold"]:
        warns.append("Kombinasi torsi x keausan tinggi — kurangi beban (risiko OSF).")
    return warns


# ============================== UI ==============================
st.set_page_config(page_title="Predictive Maintenance Dashboard",
                   page_icon="🏭", layout="wide")

# Friendly setup check for fresh clones (data & model are not committed)
_missing = []
if not (BASE / "data" / "ai4i2020.csv").exists():
    _missing.append("**data/ai4i2020.csv** — download dari "
                    "[AI4I 2020 di Kaggle](https://www.kaggle.com/datasets/stephanmatzka/"
                    "predictive-maintenance-dataset-ai4i-2020) lalu simpan di folder `data/`")
if not (BASE / "models" / "pipeline.pkl").exists():
    _missing.append("**models/pipeline.pkl** — jalankan `.venv/bin/python scripts/train_model.py` "
                    "untuk melatih model (butuh file CSV di atas)")
if _missing:
    st.title("🏭 Smart Manufacturing — Predictive Maintenance Dashboard")
    st.warning("Setup belum lengkap. Lengkapi dulu:")
    for m in _missing:
        st.write("- " + m)
    st.stop()

pipe, eda, metrics, fi, rules = load_artifacts()
df = load_data()

st.title("🏭 Smart Manufacturing — Predictive Maintenance Dashboard")
st.caption("Dataset: AI4I 2020 Predictive Maintenance (10.000 data operasi mesin milling) "
           "| Model: RandomForest, ROC-AUC {:.3f}".format(metrics["roc_auc"]))

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ringkasan Kondisi Mesin",
    "🔮 Prediksi Kegagalan",
    "🔍 Penjelasan Model (SHAP)",
    "🛠️ Jadwal Perawatan Preventif",
])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total data operasi", f"{eda['n_rows']:,}".replace(",", "."))
    c2.metric("Total kegagalan", f"{eda['n_failures']:,}".replace(",", "."))
    c3.metric("Tingkat kegagalan", f"{eda['failure_rate']:.2%}")
    c4.metric("Rata-rata keausan pahat",
              f"{eda['feature_medians']['Tool_wear_min']:.0f} mnt")

    col_a, col_b = st.columns(2)
    with col_a:
        mode_df = pd.DataFrame([
            {"Mode kegagalan": eda["mode_names"][m], "Jumlah": eda["mode_counts"][m]}
            for m in eda["mode_counts"]])
        fig = px.bar(mode_df, x="Mode kegagalan", y="Jumlah",
                     title="Kegagalan per mode",
                     color="Jumlah", color_continuous_scale="Reds")
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        type_df = pd.DataFrame([
            {"Tipe produk": t,
             "Tingkat kegagalan (%)": eda["failure_rate_by_type"][t] * 100,
             "Jumlah gagal": eda["failures_by_type"][t]}
            for t in ["L", "M", "H"]])
        fig = px.bar(type_df, x="Tipe produk", y="Tingkat kegagalan (%)",
                     title="Tingkat kegagalan per tipe produk",
                     text="Jumlah gagal", color="Tingkat kegagalan (%)",
                     color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        samp = df.sample(1500, random_state=42)
        samp["Status"] = np.where(samp["Machine_failure"] == 1, "Gagal", "Normal")
        fig = px.scatter(samp, x="Rotational_speed_rpm", y="Torque_Nm",
                         color="Status",
                         color_discrete_map={"Normal": "#1f77b4", "Gagal": "#d62728"},
                         title="Torsi vs kecepatan putar (sampel 1.500)",
                         opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        fig = px.histogram(df, x="Tool_wear_min", color="Machine_failure",
                           nbins=40, barmode="overlay",
                           title="Distribusi keausan pahat: normal vs gagal",
                           labels={"Machine_failure": "Gagal (1=ya)"},
                           opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"Performa model pada data uji (2.000 sampel): ROC-AUC "
            f"**{metrics['roc_auc']:.3f}**, Average Precision "
            f"**{metrics['avg_precision']:.3f}** — model sangat baik "
            f"membedakan kondisi normal vs berisiko gagal.")

# ---------------- TAB 2: PREDICTION ----------------
with tab2:
    st.subheader("Simulasi kondisi mesin → probabilitas kegagalan")
    med = eda["feature_medians"]
    rng = eda["feature_ranges"]
    with st.form("pred_form"):
        f1, f2, f3 = st.columns(3)
        in_type = f1.selectbox(INPUT_LABEL_ID["Type"], ["L", "M", "H"], index=0)
        in_air = f2.slider(INPUT_LABEL_ID["Air_temperature_K"],
                           float(rng["Air_temperature_K"][0]),
                           float(rng["Air_temperature_K"][1]),
                           float(med["Air_temperature_K"]), 0.1)
        in_proc = f3.slider(INPUT_LABEL_ID["Process_temperature_K"],
                            float(rng["Process_temperature_K"][0]),
                            float(rng["Process_temperature_K"][1]),
                            float(med["Process_temperature_K"]), 0.1)
        f4, f5, f6 = st.columns(3)
        in_rpm = f4.slider(INPUT_LABEL_ID["Rotational_speed_rpm"],
                           float(rng["Rotational_speed_rpm"][0]),
                           float(rng["Rotational_speed_rpm"][1]),
                           float(med["Rotational_speed_rpm"]), 1.0)
        in_torque = f5.slider(INPUT_LABEL_ID["Torque_Nm"],
                              float(rng["Torque_Nm"][0]),
                              float(rng["Torque_Nm"][1]),
                              float(med["Torque_Nm"]), 0.1)
        in_wear = f6.slider(INPUT_LABEL_ID["Tool_wear_min"],
                            float(rng["Tool_wear_min"][0]),
                            float(rng["Tool_wear_min"][1]),
                            float(med["Tool_wear_min"]), 1.0)
        submitted = st.form_submit_button("🔮 Prediksi sekarang")

    row = {"Type": in_type, "Air_temperature_K": in_air,
           "Process_temperature_K": in_proc, "Rotational_speed_rpm": in_rpm,
           "Torque_Nm": in_torque, "Tool_wear_min": in_wear,
           "Temp_diff_K": in_proc - in_air,
           "Power_W": 2 * np.pi * in_rpm * in_torque / 60.0}
    st.session_state["last_row"] = row

    if submitted or "last_proba" not in st.session_state:
        st.session_state["last_proba"] = predict_row(pipe, row)
    p = st.session_state["last_proba"]
    band, color = risk_band(p)

    g1, g2 = st.columns([1, 1])
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": "%"},
            title={"text": "Probabilitas kegagalan"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": color},
                   "steps": [{"range": [0, 25], "color": "#e8f5e9"},
                             {"range": [25, 60], "color": "#fff3e0"},
                             {"range": [60, 100], "color": "#ffebee"}],
                   "threshold": {"line": {"color": "black", "width": 3},
                                 "value": 60}}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown(f"### Kategori risiko: :{color}[{band}]")
        if band == "TINGGI":
            st.error("Rekomendasi: hentikan operasi & inspeksi segera "
                     "(≤ 24 jam). Risiko kegagalan nyata.")
        elif band == "SEDANG":
            st.warning("Rekomendasi: jadwalkan perawatan preventif "
                       "dalam ≤ 7 hari dan pantau sensor ketat.")
        else:
            st.success("Rekomendasi: lanjutkan operasi dengan monitoring rutin.")
        st.caption(f"Fitur turunan — selisih suhu: {row['Temp_diff_K']:.1f} K | "
                   f"daya: {row['Power_W']:.0f} W")
        warns = rule_warnings(row, rules)
        for w in warns:
            st.warning("⚠️ " + w)

# ---------------- TAB 3: SHAP EXPLAINABILITY ----------------
with tab3:
    st.subheader("Kenapa model bilang mesin ini berisiko?")
    st.caption("SHAP mengukur kontribusi tiap fitur terhadap probabilitas "
               "kegagalan untuk input terakhir di tab Prediksi. "
               "Batang merah = mendorong ke arah GAGAL, biru = menahan ke arah NORMAL.")
    row = st.session_state.get("last_row")
    if row is None:
        st.info("Buka tab Prediksi dan tekan tombol prediksi dulu.")
    else:
        names = transformed_feature_names(pipe)
        Xt = pipe.named_steps["pre"].transform(pd.DataFrame([row])[FEATURES])
        explainer = get_explainer(pipe)
        exp = explainer(Xt)
        try:  # shap >= 0.41: Explanation dengan dimensi kelas
            vals = exp.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]
                base = exp.base_values[:, 1] if np.ndim(exp.base_values) > 1 else exp.base_values
            else:
                base = exp.base_values
            from shap import Explanation
            exp_row = Explanation(values=vals[0], base_values=base[0]
                                  if np.ndim(base) > 0 else base,
                                  data=Xt[0], feature_names=names)
        except Exception:  # fallback API lama
            sv = explainer.shap_values(Xt)
            sv = sv[1] if isinstance(sv, list) else sv
            from shap import Explanation
            exp_row = Explanation(values=sv[0], base_values=explainer.expected_value[1]
                                  if isinstance(explainer.expected_value, (list, np.ndarray))
                                  else explainer.expected_value,
                                  data=Xt[0], feature_names=names)
        exp_row.feature_names = [LABEL_ID.get(n, n) for n in names]

        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(exp_row, max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Narasi bahasa Indonesia: 3 kontributor terbesar
        order = np.argsort(-np.abs(exp_row.values))
        st.markdown("**Faktor pendorong utama:**")
        for i in order[:3]:
            fname, val = exp_row.feature_names[i], exp_row.values[i]
            arah = "mendorong ke arah **GAGAL** 🔴" if val > 0 else \
                   "menahan ke arah **NORMAL** 🔵"
            st.write(f"- {fname}: {arah} (kontribusi {val:+.3f})")

    st.divider()
    st.subheader("Faktor terpenting secara global")
    fi_df = pd.DataFrame(fi)
    fi_df["Fitur"] = fi_df["feature"].map(lambda x: LABEL_ID.get(x, x))
    fig = px.bar(fi_df.head(8).sort_values("importance"),
                 x="importance", y="Fitur", orientation="h",
                 title="Feature importance global (RandomForest)",
                 labels={"importance": "Tingkat kepentingan"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 4: MAINTENANCE SCHEDULE ----------------
with tab4:
    st.subheader("Jadwal perawatan preventif — simulasi armada 24 mesin")
    st.caption("Diambil dari 24 sampel data historis. Probabilitas dihitung "
               "model; rekomendasi & tenggat mengikuti aturan berbasis data "
               "di bawah.")
    fleet = df.sample(24, random_state=7).reset_index(drop=True)
    rows = []
    today = date.today()
    for _, r in fleet.iterrows():
        rd = {c: r[c] for c in FEATURES}
        p = predict_row(pipe, rd)
        band, _ = risk_band(p)
        lim = rules["tool_wear_limits"].get(rd["Type"], {})
        if band == "TINGGI":
            action, due = "Inspeksi & perbaikan segera", today + timedelta(days=1)
        elif band == "SEDANG":
            action, due = "Perawatan preventif terjadwal", today + timedelta(days=7)
        elif lim and rd["Tool_wear_min"] >= lim["replace_before_min"]:
            action, due = "Ganti pahat/tool", today + timedelta(days=3)
            band = "SEDANG"
        else:
            action, due = "Monitoring rutin", today + timedelta(days=30)
        warns = rule_warnings(rd, rules)
        rows.append({
            "ID Mesin": f"M-{int(r['UDI']):05d}",
            "Tipe": rd["Type"],
            "Prob. gagal": round(p, 3),
            "Risiko": band,
            "Keausan pahat (mnt)": round(rd["Tool_wear_min"], 0),
            "Rekomendasi": action,
            "Tenggat": due.strftime("%d %b %Y"),
            "Peringatan aturan": "; ".join(warns) if warns else "—",
            "_sort": (0 if band == "TINGGI" else 1 if band == "SEDANG" else 2, -p),
        })
    sched = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")

    def risk_style(v):
        return {"TINGGI": "background-color:#ffebee", "SEDANG": "background-color:#fff3e0",
                "RENDAH": "background-color:#e8f5e9"}[v]
    st.dataframe(sched.style.map(risk_style, subset=["Risiko"]),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Aturan perawatan berbasis data")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Batas keausan pahat (ganti sebelum mencapai):**")
        for t, lim in rules["tool_wear_limits"].items():
            st.write(f"- Tipe {t}: **{lim['replace_before_min']:.0f} menit** "
                     f"(kegagalan TWF tercatat mulai {lim['min_twf_wear']:.0f} mnt)")
    with c2:
        st.markdown("**Ambang peringatan sensor:**")
        st.write(f"- Selisih suhu ≥ {rules['hdf_temp_diff_threshold']:.1f} K → "
                 f"periksa pendingin ({rules['notes']['HDF']})")
        st.write(f"- Daya di luar {rules['pwf_power_low']:.0f}–"
                 f"{rules['pwf_power_high']:.0f} W → periksa pasokan daya")
        st.write(f"- Torsi × keausan ≥ {rules['osf_strain_proxy_threshold']:.0f} → "
                 f"kurangi beban ({rules['notes']['OSF']})")

st.divider()
st.caption("Dibangun oleh Sam untuk Kia • AI4I 2020 (CC BY-NC-SA 4.0) • "
           "Model & aturan bersifat demonstrasi portofolio.")
