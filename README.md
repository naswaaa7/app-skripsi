# 🥗 Prediksi Skor Konsumsi Gizi Masyarakat Indonesia Menggunakan XGBoost

Aplikasi berbasis web yang dikembangkan menggunakan **Streamlit** dan algoritma **Extreme Gradient Boosting (XGBoost)** untuk memprediksi **Skor Konsumsi Gizi masyarakat Indonesia** pada tingkat kabupaten/kota berdasarkan **Persentase Penduduk Miskin**.

# 💻 Cara Instalasi

## Opsi 1 (Direkomendasikan)

Buka aplikasi secara langsung melalui browser tanpa perlu menginstal apa pun.

🔗 https://app-skripsi-8q6d3ccxzyvkvuugo496md.streamlit.app/

---

## Opsi 2 (Menjalankan Secara Lokal)

### 1. Clone Repository

```bash
git clone https://github.com/naswaaa7/app-skripsi.git
```

### 2. Masuk ke Folder Project

```bash
cd app-skripsi
```

### 3. Install Library

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```
---

# 🚀 Cara Menggunakan Aplikasi

### 1. Jalankan aplikasi menggunakan Streamlit.

### 2. Halaman Utama

Berisi informasi alur penggunaan sistem.

### 3. Menu Dataset

Menampilkan:

- Dataset penelitian
- Ringkasan data
- Statistik dataset

### 4. Menu Dashboard

Menampilkan visualisasi data berupa:

- Scatter Plot
- Line Chart
- Grafik distribusi data

### 5. Menu Prediksi

Langkah penggunaan:

1. Pilih Kabupaten/Kota.
2. Pilih Tahun.
3. Klik tombol **Prediksi**.
4. Sistem akan menampilkan:
   - Skor Konsumsi Gizi
   - Prediksi tahun berikutnya
   - Catatan analisis
   - Grafik aktual dan prediksi
   - Evaluasi model

### 6. Menu About

Berisi informasi mengenai penelitian dan pengembang aplikasi.

---

# 📊 Evaluasi Model

Evaluasi performa model menggunakan tiga metrik, yaitu:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 📖 Manual Book

Panduan penggunaan aplikasi dapat dilihat pada file:

**Manual_Book.pdf**

---


