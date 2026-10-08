# 🏭 Smart Manufacturing — Predictive Maintenance Dashboard

Proyek portofolio Data Analyst: analitik & predictive maintenance untuk mesin
milling berbasis **dataset AI4I 2020** (Kaggle, CC BY-NC-SA 4.0).

## Isi dashboard (4 tab)

1. **📊 Ringkasan Kondisi Mesin** — KPI (tingkat kegagalan 3,39%),
   kegagalan per mode & per tipe produk, sebaran sensor, performa model.
2. **🔮 Prediksi Kegagalan** — geser input sensor (suhu, rpm, torsi,
   keausan pahat, tipe produk) → probabilitas kegagalan + kategori
   risiko + peringatan berbasis aturan.
3. **🔍 Penjelasan Model (SHAP)** — waterfall plot: fitur apa yang
   mendorong prediksi ke arah GAGAL vs NORMAL, plus feature importance
   global.
4. **🛠️ Jadwal Perawatan Preventif** — simulasi 24 mesin: skor risiko,
   rekomendasi tindakan, tenggat, dan aturan perawatan berbasis data
   (batas keausan pahat per tipe, ambang suhu, daya).

## Model

- RandomForest (300 trees, class-balanced), fitur: 6 sensor + 2 fitur
  rekayasa (`Temp_diff_K`, `Power_W`) + one-hot tipe produk.
- Holdout 20%: **ROC-AUC 0,9715**, Average Precision 0,8602.
- Aturan perawatan diturunkan dari data (mis. ganti pahat tipe L
  sebelum 180 menit keausan).

## Dataset

Download `ai4i2020.csv` dari [Predictive Maintenance Dataset (AI4I 2020)](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
(gratis, perlu akun Kaggle) dan simpan sebagai `data/ai4i2020.csv`.

## Cara menjalankan

```bash
# 1. (sekali saja) siapkan environment & latih model
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/train_model.py

# 2. jalankan dashboard
.venv/bin/python -m streamlit run app.py
```

Lalu buka URL yang muncul di terminal (biasanya http://localhost:8501).

## Struktur

```
├── app.py                 # dashboard Streamlit
├── requirements.txt
├── data/ai4i2020.csv      # dataset (Kaggle)
├── models/
│   ├── pipeline.pkl       # model terlatih (preprocess + RF)
│   ├── metrics.json       # metrik evaluasi
│   ├── eda_summary.json   # ringkasan EDA
│   ├── feature_importance.json
│   └── maintenance_rules.json  # ambang perawatan berbasis data
└── scripts/train_model.py # training + evaluasi + aturan (reproducible)
```

Latih ulang model: `.venv/bin/python scripts/train_model.py`
