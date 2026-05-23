import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

# ==========================================
# 1. LOAD DATASET
# ==========================================
df = pd.read_csv('dataset.csv', sep=';')

target = 'Skor Konsumsi Gizi'

# ==========================================
# 2. DATA CLEANING
# ==========================================
df['Penduduk_Miskin (%)'] = pd.to_numeric(
    df['Penduduk_Miskin (%)']
    .astype(str)
    .str.replace(',', '.'),
    errors='coerce'
)

df[target] = pd.to_numeric(
    df[target]
    .astype(str)
    .str.replace(',', '.'),
    errors='coerce'
)

# Hapus data kosong
df = df.dropna(
    subset=[
        target,
        'Penduduk_Miskin (%)',
        'Provinsi'
    ]
)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
df_model = df[
    ['Penduduk_Miskin (%)', 'Provinsi', target]
].copy()

# One Hot Encoding
df_model = pd.get_dummies(
    df_model,
    columns=['Provinsi'],
    drop_first=True
)

# Pisahkan fitur dan target
X = df_model.drop(columns=[target])
y = df_model[target]

# ==========================================
# 4. SIMPAN KOLOM FITUR
# ==========================================
fitur_final = X.columns.tolist()

with open('kolom_fitur_model.pkl', 'wb') as f:
    pickle.dump(fitur_final, f)

print("✅ Kolom fitur berhasil disimpan!")

# ==========================================
# 5. TRAIN-TEST SPLIT 70:30
# ==========================================
X_train_70, X_test_70, y_train_70, y_test_70 = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Model XGBoost 70:30
model_70 = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model_70.fit(X_train_70, y_train_70)

# Prediksi
pred_70 = model_70.predict(X_test_70)

# Evaluasi
mae_70 = mean_absolute_error(y_test_70, pred_70)
mse_70 = mean_squared_error(y_test_70, pred_70)
rmse_70 = np.sqrt(mse_70)
r2_70 = r2_score(y_test_70, pred_70)

# ==========================================
# 6. TRAIN-TEST SPLIT 80:20
# ==========================================
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Model XGBoost 80:20
model_80 = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model_80.fit(X_train_80, y_train_80)

# Prediksi
pred_80 = model_80.predict(X_test_80)

# Evaluasi
mae_80 = mean_absolute_error(y_test_80, pred_80)
mse_80 = mean_squared_error(y_test_80, pred_80)
rmse_80 = np.sqrt(mse_80)
r2_80 = r2_score(y_test_80, pred_80)

# ==========================================
# 7. TAMPILKAN HASIL EVALUASI
# ==========================================
print("\n==============================")
print("HASIL EVALUASI MODEL 70:30")
print("==============================")
print(f"MAE   : {mae_70:.2f}")
print(f"MSE   : {mse_70:.2f}")
print(f"RMSE  : {rmse_70:.2f}")
print(f"R²    : {r2_70:.2f}")

print("\n==============================")
print("HASIL EVALUASI MODEL 80:20")
print("==============================")
print(f"MAE   : {mae_80:.2f}")
print(f"MSE   : {mse_80:.2f}")
print(f"RMSE  : {rmse_80:.2f}")
print(f"R²    : {r2_80:.2f}")

# ==========================================
# 8. SIMPAN SEMUA HASIL EVALUASI
# ==========================================
evaluasi_model = {

    '70:30': {
        'mae': mae_70,
        'mse': mse_70,
        'rmse': rmse_70,
        'r2': r2_70
    },

    '80:20': {
        'mae': mae_80,
        'mse': mse_80,
        'rmse': rmse_80,
        'r2': r2_80
    }
}

with open('evaluasi_model.pkl', 'wb') as f:
    pickle.dump(evaluasi_model, f)

print("✅ Evaluasi model berhasil disimpan!")

# ==========================================
# 9. SIMPAN DATA VISUALISASI
# ==========================================
visualisasi_data = {

    '70:30': {
        'y_test': y_test_70.tolist(),
        'preds': pred_70.tolist()
    },

    '80:20': {
        'y_test': y_test_80.tolist(),
        'preds': pred_80.tolist()
    }
}

with open('visualisasi_model.pkl', 'wb') as f:
    pickle.dump(visualisasi_data, f)

print("✅ Data visualisasi berhasil disimpan!")

# ==========================================
# 10. PILIH MODEL TERBAIK
# ==========================================
if r2_80 > r2_70:

    model_terbaik = model_80
    split_terbaik = "80:20"

else:

    model_terbaik = model_70
    split_terbaik = "70:30"

print(f"\n✅ Model terbaik menggunakan split {split_terbaik}")

# ==========================================
# 11. SIMPAN MODEL TERBAIK
# ==========================================
with open('model_xgboost_skripsi.pkl', 'wb') as f:
    pickle.dump(model_terbaik, f)

print("✅ Model XGBoost terbaik berhasil disimpan!")