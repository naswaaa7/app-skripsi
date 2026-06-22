import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================
df = pd.read_csv("dataset.csv", sep=";")

target = "Skor Konsumsi Gizi"

# ==========================================
# 2. PENYESUAIAN NAMA KOLOM
# ==========================================
df = df.rename(columns={
    "Kabupaten_Kota": "Kabupaten/Kota",
    "Penduduk_Miskin (%)": "Penduduk_Miskin (%)"
})

# ==========================================
# 3. DATA CLEANING
# ==========================================
df["Penduduk_Miskin (%)"] = pd.to_numeric(
    df["Penduduk_Miskin (%)"].astype(str).str.replace(",", "."),
    errors="coerce"
)

df[target] = pd.to_numeric(
    df[target].astype(str).str.replace(",", "."),
    errors="coerce"
)

df["Tahun"] = pd.to_numeric(
    df["Tahun"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Provinsi",
        "Kabupaten/Kota",
        "Tahun",
        "Penduduk_Miskin (%)",
        target
    ]
)

df["Tahun"] = df["Tahun"].astype(int)

# ==========================================
# 4. ANALISIS KORELASI NASIONAL
# ==========================================
korelasi_miskin_skor = df["Penduduk_Miskin (%)"].corr(
    df[target],
    method="pearson"
)

print("\n==============================")
print("HASIL ANALISIS KORELASI")
print("==============================")
print(f"Korelasi Penduduk Miskin terhadap Skor Konsumsi Gizi: {korelasi_miskin_skor:.3f}")

hasil_korelasi = {
    "Variabel X": "Penduduk_Miskin (%)",
    "Variabel Y": target,
    "Metode": "Pearson",
    "Nilai Korelasi": korelasi_miskin_skor
}

with open("hasil_korelasi.pkl", "wb") as f:
    pickle.dump(hasil_korelasi, f)

# ==========================================
# 5. EVALUASI MODEL PER PROVINSI
# Data latih = 2021-2022
# Data uji   = 2023
# ==========================================
model_evaluasi_per_provinsi = {}
fitur_per_provinsi = {}
evaluasi_per_provinsi = {}
visualisasi_per_provinsi = {}

semua_y_test = []
semua_prediksi = []

for provinsi in sorted(df["Provinsi"].unique()):

    df_prov = df[df["Provinsi"] == provinsi].copy()

    data_train = df_prov[df_prov["Tahun"].isin([2021, 2022])].copy()
    data_test = df_prov[df_prov["Tahun"] == 2023].copy()

    if data_train.empty or data_test.empty:
        print(f"\nProvinsi {provinsi} dilewati karena data train/test tidak lengkap.")
        continue

    # Feature engineering per provinsi
    df_model = df_prov[
        [
            "Tahun",
            "Penduduk_Miskin (%)",
            "Kabupaten/Kota",
            target
        ]
    ].copy()

    df_model = pd.get_dummies(
        df_model,
        columns=["Kabupaten/Kota"],
        drop_first=True
    )

    data_train_model = df_model[df_model["Tahun"].isin([2021, 2022])]
    data_test_model = df_model[df_model["Tahun"] == 2023]

    X_train = data_train_model.drop(columns=[target])
    y_train = data_train_model[target]

    X_test = data_test_model.drop(columns=[target])
    y_test = data_test_model[target]

    fitur_final = X_train.columns.tolist()

    model_eval = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model_eval.fit(X_train, y_train)

    pred = model_eval.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)

    if len(y_test) > 1:
        r2 = r2_score(y_test, pred)
    else:
        r2 = None

    model_evaluasi_per_provinsi[provinsi] = model_eval
    fitur_per_provinsi[provinsi] = fitur_final

    evaluasi_per_provinsi[provinsi] = {
        "data_latih": "2021-2022",
        "data_uji": "2023",
        "jumlah_data_train": len(data_train),
        "jumlah_data_test": len(data_test),
        "jumlah_kabupaten_kota": df_prov["Kabupaten/Kota"].nunique(),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    visualisasi_per_provinsi[provinsi] = {
        "kabupaten_kota": data_test["Kabupaten/Kota"].tolist(),
        "y_test": y_test.tolist(),
        "preds": pred.tolist()
    }

    semua_y_test.extend(y_test.tolist())
    semua_prediksi.extend(pred.tolist())

    print("\n==============================")
    print(f"EVALUASI MODEL PROVINSI: {provinsi}")
    print("==============================")
    print("Data latih          : 2021-2022")
    print("Data uji            : 2023")
    print(f"Jumlah data latih   : {len(data_train)}")
    print(f"Jumlah data uji     : {len(data_test)}")
    print(f"Jumlah kab/kota     : {df_prov['Kabupaten/Kota'].nunique()}")
    print(f"MAE                 : {mae:.2f}")
    print(f"RMSE                : {rmse:.2f}")

    if r2 is not None:
        print(f"R²                  : {r2:.2f}")
    else:
        print("R²                  : Tidak dapat dihitung karena data uji hanya 1")

# ==========================================
# 6. EVALUASI GABUNGAN SELURUH MODEL PROVINSI
# ==========================================
mae_total = mean_absolute_error(semua_y_test, semua_prediksi)
mse_total = mean_squared_error(semua_y_test, semua_prediksi)
rmse_total = np.sqrt(mse_total)
r2_total = r2_score(semua_y_test, semua_prediksi)

evaluasi_total = {
    "data_latih": "2021-2022",
    "data_uji": "2023",
    "mae": mae_total,
    "mse": mse_total,
    "rmse": rmse_total,
    "r2": r2_total
}

print("\n==============================")
print("EVALUASI GABUNGAN MODEL PER PROVINSI")
print("==============================")
print("Data latih : 2021-2022")
print("Data uji   : 2023")
print(f"MAE        : {mae_total:.2f}")
print(f"RMSE       : {rmse_total:.2f}")
print(f"R²         : {r2_total:.2f}")

# ==========================================
# 7. EVALUASI MODEL PER TAHUN PER PROVINSI
# Tahun uji yang tersedia aktualnya = 2022 dan 2023
# ==========================================
evaluasi_per_tahun_per_provinsi = {}
visualisasi_per_tahun_per_provinsi = {}

daftar_tahun_uji = [2022, 2023]

for tahun_uji in daftar_tahun_uji:

    evaluasi_per_tahun_per_provinsi[tahun_uji] = {}
    visualisasi_per_tahun_per_provinsi[tahun_uji] = {}

    print("\n======================================")
    print(f"EVALUASI MODEL PER TAHUN: {tahun_uji}")
    print("======================================")

    semua_y_test_tahun = []
    semua_prediksi_tahun = []

    for provinsi in sorted(df["Provinsi"].unique()):

        df_prov = df[df["Provinsi"] == provinsi].copy()

        # Data latih memakai tahun sebelum tahun uji
        data_train = df_prov[df_prov["Tahun"] < tahun_uji].copy()

        # Data uji memakai tahun yang sedang diuji
        data_test = df_prov[df_prov["Tahun"] == tahun_uji].copy()

        if data_train.empty or data_test.empty:
            print(f"Provinsi {provinsi} dilewati karena data train/test tahun {tahun_uji} tidak lengkap.")
            continue

        df_model = df_prov[
            [
                "Tahun",
                "Penduduk_Miskin (%)",
                "Kabupaten/Kota",
                target
            ]
        ].copy()

        df_model = pd.get_dummies(
            df_model,
            columns=["Kabupaten/Kota"],
            drop_first=True
        )

        data_train_model = df_model[df_model["Tahun"] < tahun_uji]
        data_test_model = df_model[df_model["Tahun"] == tahun_uji]

        X_train = data_train_model.drop(columns=[target])
        y_train = data_train_model[target]

        X_test = data_test_model.drop(columns=[target])
        y_test = data_test_model[target]

        model_eval_tahun = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        model_eval_tahun.fit(X_train, y_train)

        pred = model_eval_tahun.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)

        if len(y_test) > 1:
            r2 = r2_score(y_test, pred)
        else:
            r2 = None

        if tahun_uji == 2022:
            data_latih_text = "2021"
        else:
            data_latih_text = f"2021-{tahun_uji - 1}"

        evaluasi_per_tahun_per_provinsi[tahun_uji][provinsi] = {
            "data_latih": data_latih_text,
            "data_uji": str(tahun_uji),
            "jumlah_data_train": len(data_train),
            "jumlah_data_test": len(data_test),
            "jumlah_kabupaten_kota": df_prov["Kabupaten/Kota"].nunique(),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }

        visualisasi_per_tahun_per_provinsi[tahun_uji][provinsi] = {
            "kabupaten_kota": data_test["Kabupaten/Kota"].tolist(),
            "y_test": y_test.tolist(),
            "preds": pred.tolist()
        }

        semua_y_test_tahun.extend(y_test.tolist())
        semua_prediksi_tahun.extend(pred.tolist())

        print("\n------------------------------")
        print(f"Provinsi       : {provinsi}")
        print(f"Data latih     : {data_latih_text}")
        print(f"Data uji       : {tahun_uji}")
        print(f"Jumlah train   : {len(data_train)}")
        print(f"Jumlah test    : {len(data_test)}")
        print(f"MAE            : {mae:.2f}")
        print(f"RMSE           : {rmse:.2f}")

        if r2 is not None:
            print(f"R²             : {r2:.2f}")
        else:
            print("R²             : Tidak dapat dihitung")

    # Evaluasi gabungan per tahun
    if len(semua_y_test_tahun) > 0:
        mae_tahun_total = mean_absolute_error(semua_y_test_tahun, semua_prediksi_tahun)
        mse_tahun_total = mean_squared_error(semua_y_test_tahun, semua_prediksi_tahun)
        rmse_tahun_total = np.sqrt(mse_tahun_total)
        r2_tahun_total = r2_score(semua_y_test_tahun, semua_prediksi_tahun)

        evaluasi_per_tahun_per_provinsi[tahun_uji]["TOTAL"] = {
            "data_latih": "2021" if tahun_uji == 2022 else f"2021-{tahun_uji - 1}",
            "data_uji": str(tahun_uji),
            "mae": mae_tahun_total,
            "mse": mse_tahun_total,
            "rmse": rmse_tahun_total,
            "r2": r2_tahun_total
        }

        print("\n==============================")
        print(f"EVALUASI GABUNGAN TAHUN {tahun_uji}")
        print("==============================")
        print(f"MAE  : {mae_tahun_total:.2f}")
        print(f"RMSE : {rmse_tahun_total:.2f}")
        print(f"R²   : {r2_tahun_total:.2f}")

# ==========================================
# 8. TRAINING MODEL FINAL PER PROVINSI
# Data latih final = 2021-2023
# Digunakan untuk prediksi 2024
# ==========================================
model_final_per_provinsi = {}
fitur_final_per_provinsi = {}

for provinsi in sorted(df["Provinsi"].unique()):

    df_prov = df[df["Provinsi"] == provinsi].copy()

    data_train_final = df_prov[df_prov["Tahun"].isin([2021, 2022, 2023])].copy()

    if data_train_final.empty:
        print(f"\nProvinsi {provinsi} dilewati karena data final tidak tersedia.")
        continue

    df_model_final = data_train_final[
        [
            "Tahun",
            "Penduduk_Miskin (%)",
            "Kabupaten/Kota",
            target
        ]
    ].copy()

    df_model_final = pd.get_dummies(
        df_model_final,
        columns=["Kabupaten/Kota"],
        drop_first=True
    )

    X_final = df_model_final.drop(columns=[target])
    y_final = df_model_final[target]

    fitur_final = X_final.columns.tolist()

    model_final = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model_final.fit(X_final, y_final)

    model_final_per_provinsi[provinsi] = model_final
    fitur_final_per_provinsi[provinsi] = fitur_final

    print(f"✅ Model final Provinsi {provinsi} dilatih dengan data 2021-2023")

# ==========================================
# 9. SIMPAN MODEL DAN HASIL EVALUASI
# ==========================================

# Model ini untuk evaluasi aktual 2023 vs prediksi 2023
with open("model_evaluasi_per_provinsi.pkl", "wb") as f:
    pickle.dump(model_evaluasi_per_provinsi, f)

# Model ini untuk prediksi final tahun 2024
with open("model_per_provinsi.pkl", "wb") as f:
    pickle.dump(model_final_per_provinsi, f)

# Fitur final untuk model prediksi 2024
with open("fitur_per_provinsi.pkl", "wb") as f:
    pickle.dump(fitur_final_per_provinsi, f)

# Evaluasi utama 2021-2022 -> 2023
with open("evaluasi_per_provinsi.pkl", "wb") as f:
    pickle.dump(evaluasi_per_provinsi, f)

with open("visualisasi_per_provinsi.pkl", "wb") as f:
    pickle.dump(visualisasi_per_provinsi, f)

with open("evaluasi_total_model_per_provinsi.pkl", "wb") as f:
    pickle.dump(evaluasi_total, f)

# Evaluasi tambahan per tahun
with open("evaluasi_per_tahun_per_provinsi.pkl", "wb") as f:
    pickle.dump(evaluasi_per_tahun_per_provinsi, f)

with open("visualisasi_per_tahun_per_provinsi.pkl", "wb") as f:
    pickle.dump(visualisasi_per_tahun_per_provinsi, f)

print("\n✅ Semua model evaluasi, model final, dan evaluasi per tahun berhasil disimpan!")