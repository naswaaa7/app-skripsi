import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle
import xgboost as xgb
import base64

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================

st.set_page_config(
    page_title="Sistem Prediksi Konsumsi Gizi",
    layout="wide"
)

# ==========================================
# 2. DESAIN TAMPILAN
# ==========================================

def set_design_theme():
    st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');
     
    .stApp {
        background: linear-gradient(135deg, #F8FFF4 0%, #FFFFFF 45%, #FFF8D8 100%);
        color: #2F3E2F;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F1F8E9 0%, #FFFDE7 100%);
        border-right: 1px solid #DDECC8;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #2E5E3E !important;
    }

    h1 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #2E5E3E !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #3F6F44 !important;
        font-weight: 700 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        border: 1.5px solid #DDECC8;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(82, 120, 72, 0.10);
    }

    div.stButton > button:first-child {
        background: #7BAE6A;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1rem;
        font-weight: 700;
        box-shadow: 0 6px 14px rgba(91, 125, 77, 0.18);
    }

    div.stButton > button:first-child:hover {
        background: #6A9E5B;
        color: white;
        transform: translateY(-1px);
    }

    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #DDECC8;
    }

    hr {
        border: none;
        height: 1px;
        background: #DDECC8;
        margin: 1.5rem 0;
    }

    .hero-box {
    background-color: #FFFFFF;
    padding: 34px 36px 30px 36px;
    border-radius: 24px;
    border: 2px solid #D9E7C6;
    box-shadow: 0 12px 28px rgba(82, 120, 72, 0.14);
    margin-bottom: 0;
    }

    .hero-subtitle {
        color: #65745F;
        font-size: 16px;
        text-align: center;
        margin-top: 12px;
        line-height: 1.7;
    }

    .spacer-small {
        height: 28px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 2px solid #D9E7C6 !important;
        border-radius: 22px !important;
        box-shadow: 0 10px 24px rgba(82, 120, 72, 0.12) !important;
        padding: 22px 26px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border-radius: 22px !important;
    }

    div[data-testid="stMarkdownContainer"] {
        color: #445C3C;
    }
    </style>
    """, unsafe_allow_html=True)

set_design_theme()

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_main_page_background(image_path):
    encoded_image = get_base64_image(image_path)
    st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        [data-testid="stToolbar"] {{
            right: 2rem;
        }}

        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.10);
            z-index: 0;
        }}

        .block-container {{
            position: relative;
            z-index: 1;
        }}
        </style>
    """, unsafe_allow_html=True)

def reset_default_background():
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #F8FFF4 0%, #FFFFFF 45%, #FFF8D8 100%);
            background-image: none;
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI UNTUK LOAD DATA DAN MODEL
# ==========================================

@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv('dataset.csv', sep=';')
        df = df.rename(columns={
            'Kabupaten_Kota': 'Kabupaten/Kota',
            'Penduduk_Miskin (%)': 'Persentase Miskin (%)'
        })

        for col in ['Persentase Miskin (%)', 'Skor Konsumsi Gizi']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.'),
                    errors='coerce'
                )
        return df

    except Exception:
        return pd.DataFrame(
            columns=[
                "Kabupaten/Kota",
                "Tahun",
                "Persentase Miskin (%)",
                "Skor Konsumsi Gizi"
            ]
        )

df_dummy = load_default_data()

@st.cache_resource
def load_model_files():
    try:
        with open('model_xgboost_skripsi.pkl', 'rb') as f:
            m = pickle.load(f)

        with open('kolom_fitur_model.pkl', 'rb') as f:
            c = pickle.load(f)

        with open('evaluasi_model.pkl', 'rb') as f:
            e = pickle.load(f)

        with open('visualisasi_model.pkl', 'rb') as f:
            v = pickle.load(f)

        return m, c, e, v

    except Exception:
        return None, None, None, None

xgb_model, fitur_model, evaluasi_model, visualisasi_model = load_model_files()

# ==========================================
# 4. SESSION STATE AWAL
# ==========================================

if 'data_valid' not in st.session_state:
    st.session_state['data_valid'] = True

# Mapping Provinsi -> Pulau (untuk filter Pulau di Dashboard)
MAPPING_PULAU = {
    'Aceh': 'Sumatera', 'Sumatera Utara': 'Sumatera', 'Sumatera Barat': 'Sumatera',
    'Riau': 'Sumatera', 'Jambi': 'Sumatera', 'Sumatera Selatan': 'Sumatera',
    'Bengkulu': 'Sumatera', 'Lampung': 'Sumatera', 'Kep. Bangka Belitung': 'Sumatera',
    'Kep. Riau': 'Sumatera',

    'DKI Jakarta': 'Jawa', 'Jawa Barat': 'Jawa', 'Jawa Tengah': 'Jawa',
    'DI Yogyakarta': 'Jawa', 'Jawa Timur': 'Jawa', 'Banten': 'Jawa',

    'Kalimantan Barat': 'Kalimantan', 'Kalimantan Tengah': 'Kalimantan',
    'Kalimantan Selatan': 'Kalimantan', 'Kalimantan Timur': 'Kalimantan',
    'Kalimantan Utara': 'Kalimantan',

    'Sulawesi Utara': 'Sulawesi', 'Sulawesi Tengah': 'Sulawesi',
    'Sulawesi Selatan': 'Sulawesi', 'Sulawesi Tenggara': 'Sulawesi',
    'Gorontalo': 'Sulawesi', 'Sulawesi Barat': 'Sulawesi',

    'Papua': 'Papua', 'Papua Barat': 'Papua',
}

def tambah_kolom_pulau(df):
    df = df.copy()

    if 'Provinsi' in df.columns:
        df['Pulau'] = df['Provinsi'].map(MAPPING_PULAU)
        df['Pulau'] = df['Pulau'].fillna('Lainnya')

    return df

if 'df_user_upload' not in st.session_state:
    st.session_state['df_user_upload'] = None

if 'df_aktif' not in st.session_state:
    st.session_state['df_aktif'] = tambah_kolom_pulau(df_dummy)

# ==========================================
# 5. SIDEBAR NAVIGASI
# ==========================================

st.sidebar.title("Navigation Menu")

menu = [
    "Main Page",
    "Dataset",
    "Dashboard Visualisasi",
    "Prediksi",
    "About"
]

pilihan = st.sidebar.radio("Silakan pilih halaman:", menu)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sistem Prediksi Skor Konsumsi Gizi Berdasarkan Tingkat Kemiskinan menggunakan XGBoost."
)

# ==========================================
# 6. LOGIKA HALAMAN
# ==========================================

reset_default_background()

# --- HALAMAN MAIN PAGE ---
if pilihan == "Main Page":
    set_main_page_background("background_mainpage.png")

    st.markdown("""
    <div class="hero-box">
        <h1 style="text-align:center;">
            Prediksi Konsumsi Gizi Masyarakat Berdasarkan Persentase Penduduk Miskin dengan Algoritma XGBoost
        </h1>
        <p class="hero-subtitle">
            Aplikasi berbasis Streamlit dengan algoritma XGBoost untuk membantu analisis hubungan antara persentase penduduk miskin dan skor konsumsi gizi masyarakat.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)

    st.markdown('<div class="spacer-small"></div>', unsafe_allow_html=True)

    st.html("""
    <div style="
        background-color: #FFFFFF;
        border: 2px solid #D9E7C6;
        border-radius: 22px;
        padding: 26px 30px;
        box-shadow: 0 10px 24px rgba(82, 120, 72, 0.14);
        color: #445C3C;
    ">
        <h3 style="
            color: #3F6F44;
            margin-top: 0;
            margin-bottom: 22px;
            font-size: 24px;
            font-weight: 800;
        ">
            Alur Penggunaan Sistem
        </h3>

        <p style="
            color: #2E5E3E;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 6px;
        ">
            Langkah 1: Tentukan Basis Data
        </p>
        <p style="
            font-size: 16px;
            line-height: 1.8;
            margin-bottom: 20px;
        ">
            Mulai dengan membuka menu Dataset pada panel navigasi di sebelah kiri.
            Pengguna dapat menggunakan dataset bawaan sistem atau mengunggah file data penelitian
            dalam format CSV atau Excel.
        </p>

        <p style="
            color: #2E5E3E;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 6px;
        ">
            Langkah 2: Eksplorasi Visualisasi
        </p>
        <p style="
            font-size: 16px;
            line-height: 1.8;
            margin-bottom: 20px;
        ">
            Setelah data berhasil dimuat, pengguna dapat membuka menu Dashboard Visualisasi
            untuk melihat pola sebaran, tren tahunan, serta hubungan antara tingkat kemiskinan
            dan capaian konsumsi gizi.
        </p>

        <p style="
            color: #2E5E3E;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 6px;
        ">
            Langkah 3: Jalankan Prediksi
        </p>
        <p style="
            font-size: 16px;
            line-height: 1.8;
            margin-bottom: 0;
        ">
            Pada menu Prediksi, pengguna dapat memilih kabupaten/kota dan tahun observasi.
            Sistem akan menggunakan model XGBoost untuk menghasilkan prediksi skor konsumsi gizi
            tahun berikutnya.
        </p>
    </div>
    """)

# --- HALAMAN DATASET ---
elif pilihan == "Dataset":
    st.title("Data Penelitian")
    st.write(
        "Tentukan sumber data yang akan dijadikan basis acuan untuk visualisasi dan prediksi pada sistem ini."
    )

    st.warning("""
    Perhatian sebelum unggah data.

    Pastikan file Excel atau CSV memiliki kolom dengan nama sebagai berikut:
    1. Kabupaten/Kota
    2. Tahun
    3. Penduduk Miskin (%)
    4. Skor Konsumsi Gizi
    """)

    st.divider()

    kolom_wajib_mentah = [
        'Kabupaten/Kota',
        'Tahun',
        'Penduduk Miskin (%)',
        'Skor Konsumsi Gizi'
    ]

    if st.session_state['df_user_upload'] is not None:
        df_mentah = st.session_state['df_user_upload']
        kolom_hilang = [kol for kol in kolom_wajib_mentah if kol not in df_mentah.columns]

        if len(kolom_hilang) > 0:
            st.error(
                f"Akses modul diblokir. Dataset unggahan tidak memiliki kolom: {', '.join(kolom_hilang)}"
            )
            st.info(
                "Solusi: hapus dataset unggahan untuk kembali menggunakan dataset bawaan sistem."
            )

            if st.button("Hapus Dataset Saya dan Gunakan Dataset Default"):
                st.session_state['df_user_upload'] = None
                st.session_state['df_aktif'] = tambah_kolom_pulau(df_dummy)
                st.session_state['data_valid'] = True
                st.rerun()

            st.session_state['data_valid'] = False

        else:
            st.success("Dataset unggahan terdeteksi dan tersimpan di sistem.")

            sumber = st.radio(
                "Pilih Data yang Digunakan:",
                ("Dataset Unggahan Saya", "Gunakan Dataset Bawaan Sistem"),
                horizontal=True
            )

            if sumber == "Dataset Unggahan Saya":
                df_bersih = df_mentah.copy()
                df_bersih = df_bersih.rename(columns={
                    'Kabupaten_Kota': 'Kabupaten/Kota',
                    'Penduduk Miskin (%)': 'Persentase Miskin (%)'
                })

                df_bersih['Persentase Miskin (%)'] = pd.to_numeric(
                    df_bersih['Persentase Miskin (%)'].astype(str).str.replace(',', '.'),
                    errors='coerce'
                )

                df_bersih['Skor Konsumsi Gizi'] = pd.to_numeric(
                    df_bersih['Skor Konsumsi Gizi'].astype(str).str.replace(',', '.'),
                    errors='coerce'
                )

                df_bersih = tambah_kolom_pulau(df_bersih)

                st.session_state['df_aktif'] = df_bersih
                st.session_state['data_valid'] = True

            else:
                st.session_state['df_aktif'] = tambah_kolom_pulau(df_dummy)
                st.session_state['data_valid'] = True

            if st.button("Hapus File Unggahan"):
                st.session_state['df_user_upload'] = None
                st.rerun()

    else:
        pilihan_sumber_data = st.radio(
            "Pilih Sumber Data:",
            ("Gunakan Dataset Bawaan Sistem", "Unggah Dataset Sendiri"),
            horizontal=True
        )

        if pilihan_sumber_data == "Gunakan Dataset Bawaan Sistem":
            st.session_state['df_aktif'] = tambah_kolom_pulau(df_dummy)
            st.session_state['data_valid'] = True
            st.success("Sistem saat ini menggunakan dataset bawaan.")

        elif pilihan_sumber_data == "Unggah Dataset Sendiri":
            st.info("Silakan unggah file Anda. Pastikan format kolom sesuai aturan di atas.")
            uploaded_file = st.file_uploader("Upload format CSV atau Excel:", type=["csv", "xlsx"])

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        try:
                            df_m = pd.read_csv(uploaded_file, sep=';')
                            if len(df_m.columns) < 2:
                                uploaded_file.seek(0)
                                df_m = pd.read_csv(uploaded_file, sep=',')
                        except Exception:
                            uploaded_file.seek(0)
                            df_m = pd.read_csv(uploaded_file)
                    else:
                        df_m = pd.read_excel(uploaded_file)

                    st.session_state['df_user_upload'] = df_m
                    st.rerun()

                except Exception as e:
                    st.error(f"Gagal memproses file: {e}")
                    st.session_state['data_valid'] = False

    if st.session_state.get('data_valid', False):
        df_tampil = st.session_state.get('df_aktif', pd.DataFrame())

        if not df_tampil.empty:
            st.markdown("### Ringkasan Informasi Data")

            kol1, kol2, kol3 = st.columns(3)

            kol1.metric("Total Data", len(df_tampil))

            if 'Kabupaten/Kota' in df_tampil.columns:
                kol2.metric("Jumlah Wilayah", df_tampil['Kabupaten/Kota'].nunique())

            if 'Skor Konsumsi Gizi' in df_tampil.columns:
                kol3.metric(
                    "Rata-rata Skor Gizi",
                    f"{df_tampil['Skor Konsumsi Gizi'].mean():.2f}"
                )

            st.dataframe(df_tampil, use_container_width=True, hide_index=True)

# --- HALAMAN DASHBOARD VISUALISASI ---
elif pilihan == "Dashboard Visualisasi":
    st.title("Dashboard Visualisasi")
    
    if not st.session_state.get('data_valid', False):
        st.error("**AKSES DIBLOKIR:** Visualisasi tidak dapat ditampilkan karena dataset belum dipilih atau file yang diunggah tidak valid.")
        st.warning("Silakan kembali ke menu Dataset dan pilih sumber data atau unggah file dengan kategori yang benar.")
    else:
        st.write("Visualisasi ini membantu peneliti melihat pola persebaran dan hubungan data sebelum diproses oleh model.")
        df_dash = st.session_state.get('df_aktif', df_dummy).copy()
        df_dash = tambah_kolom_pulau(df_dash)

        st.markdown("### Filter Data")

        kol_filter1, kol_filter2, kol_filter3 = st.columns(3)
        with kol_filter1:
            if 'Pulau' in df_dash.columns:
                list_pulau = ["Jawa", "Sumatera", "Sulawesi", "Kalimantan", "Papua"]

                pilih_pulau = st.multiselect(
                    "Pilih Pulau:",
                    list_pulau,
                    default=list_pulau
                )

                df_terpulau = df_dash[df_dash['Pulau'].isin(pilih_pulau)]

            else:
                pilih_pulau = []
                df_terpulau = df_dash
        with kol_filter2:
            list_kabkota = sorted(df_terpulau['Kabupaten/Kota'].unique().tolist())
            default_kabkota = list_kabkota[:5] if len(list_kabkota) > 5 else list_kabkota
            pilih_wilayah = st.multiselect("Pilih Kabupaten/Kota:", list_kabkota, default=default_kabkota)
        with kol_filter3:
            list_tahun = sorted(df_terpulau['Tahun'].unique().tolist())
            pilih_tahun = st.multiselect("Pilih Tahun:", list_tahun, default=list_tahun)

        df_filtered = df_terpulau[(df_terpulau['Kabupaten/Kota'].isin(pilih_wilayah)) & (df_terpulau['Tahun'].isin(pilih_tahun))]

        if df_filtered.empty:
            st.warning("Data tidak tersedia. Silakan ubah filter pulau, wilayah, atau tahun di atas.")
        else:
            st.divider()
            st.subheader("1. Hubungan Kemiskinan dengan Skor Gizi")
            st.caption("Setiap titik mewakili satu wilayah pada tahun tertentu, perhatikan posisi titiknya untuk membaca hubungan antara tingkat kemiskinan dan skor konsumsi gizi di seluruh wilayah yang dipilih.")
            df_scatter = df_filtered.reset_index(drop=True).copy()
            df_scatter['Kabupaten/Kota'] = df_scatter['Kabupaten/Kota'].astype(str)
            df_scatter['Label Tahun'] = df_scatter['Tahun'].astype(str)
            fig_scatter = px.scatter(
                df_scatter,
                x="Persentase Miskin (%)",
                y="Skor Konsumsi Gizi",
                color="Kabupaten/Kota",
                hover_data=["Kabupaten/Kota", "Tahun", "Persentase Miskin (%)", "Skor Konsumsi Gizi"],
                height=450,
            )
            fig_scatter.update_traces(textposition="top center", marker=dict(size=11))
            fig_scatter.update_layout(
                xaxis_title="Persentase Penduduk Miskin (%)",
                yaxis_title="Skor Konsumsi Gizi",
                legend_title_text="Kabupaten/Kota",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            with st.expander("Cara Membaca Grafik Ini"):
                st.write("""
                - Sumbu bawah menunjukkan persentase penduduk miskin.
                Semakin ke kanan, berarti tingkat kemiskinan semakin tinggi.
                - Sumbu samping menunjukkan skor konsumsi gizi.
                Semakin ke atas, berarti kualitas konsumsi gizi semakin baik.
                - Setiap titik mewakili satu kabupaten/kota pada tahun tertentu.
                - Jika titik berada di kanan bawah, artinya wilayah tersebut memiliki tingkat kemiskinan tinggi tetapi skor konsumsi gizinya rendah. 
                Sebaliknya, titik di kiri atas menunjukkan kondisi gizi yang lebih baik dengan tingkat kemiskinan yang lebih rendah.
                - Grafik ini membantu melihat apakah kenaikan tingkat kemiskinan berpengaruh terhadap penurunan skor konsumsi gizi masyarakat.
                """)

            st.divider()
            st.subheader("2. Perkembangan Gizi Tahunan")
            st.caption("Garis memperlihatkan perkembangan skor konsumsi gizi setiap kabupaten/kota dari tahun ke tahun.")
            df_trend = (
                df_filtered.groupby(['Tahun', 'Kabupaten/Kota'], as_index=False)['Skor Konsumsi Gizi']
                .mean()
                .sort_values('Tahun')
                .reset_index(drop=True)
            )
            df_trend['Kabupaten/Kota'] = df_trend['Kabupaten/Kota'].astype(str)
            if df_trend['Tahun'].nunique() < 2:
                st.info("Pilih minimal 2 tahun observasi agar garis perkembangan dapat terbentuk.")
            fig_trend = px.line(
                df_trend,
                x="Tahun",
                y="Skor Konsumsi Gizi",
                color="Kabupaten/Kota",
                markers=True,
                labels={
                    "Tahun" : "Tahun Observasi",
                    "Skor Konsumsi Gizi" : "Skor Konsumsi Gizi",
                    "Kabupaten/Kota" : "Kabupaten/Kota",
                },
                height=450,
            )
            fig_trend.update_layout(
                xaxis_title="Tahun Observasi",
                yaxis_title="Skor Konsumsi Gizi",
                legend_title_text="Kabupaten/Kota",
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{len(df_filtered)}_{'-'.join(map(str, pilih_tahun))}")
            with st.expander("Cara Membaca Grafik Ini"):
                st.write("""
                Grafik garis di atas menunjukkan perkembangan skor konsumsi gizi setiap wilayah dari tahun ke tahun.

                - Sumbu bawah menunjukkan tahun observasi.
                - Sumbu samping menunjukkan skor konsumsi gizi.
                - Setiap garis mewakili satu kabupaten/kota.
                - Garis yang naik menunjukkan bahwa skor konsumsi gizi di wilayah tersebut mengalami peningkatan.
                - Garis yang turun menunjukkan adanya penurunan skor konsumsi gizi.
                - Semakin stabil dan tinggi posisi garis, maka kondisi konsumsi gizi wilayah tersebut cenderung lebih baik.
                - Grafik ini membantu melihat wilayah mana yang mengalami peningkatan atau penurunan kualitas konsumsi gizi setiap tahunnya.
                """)

            st.divider()
            st.subheader("3. Perbandingan Skor Antar Wilayah")
            df_bar = df_filtered.reset_index(drop=True).copy()
            df_bar['Kabupaten/Kota'] = df_bar['Kabupaten/Kota'].astype(str)
            df_bar['Tahun'] = df_bar['Tahun'].astype(str)
            fig_bar = px.bar(
                df_bar,
                x="Kabupaten/Kota",
                y="Skor Konsumsi Gizi",
                color="Tahun",
                barmode="group",
                labels={
                    "Kabupaten/Kota" : "Kabupaten/Kota",
                    "Skor Konsumsi Gizi" : "Skor Konsumsi Gizi",
                    "Tahun": "Tahun",
                },
                height=450,
            )
            fig_bar.update_layout(
                xaxis_title="Kabupaten/Kota",
                yaxis_title="Skor Konsumsi Gizi",
                legend_title_text="Tahun",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            with st.expander("Cara Membaca Grafik Ini"):
                st.write("""
                Grafik batang di atas digunakan untuk membandingkan skor konsumsi gizi antar wilayah pada setiap tahun.

                - Sumbu bawah menunjukkan kabupaten/kota.
                - Sumbu samping menunjukkan skor konsumsi gizi.
                - Setiap warna batang menunjukkan tahun yang berbeda.
                - Semakin tinggi batang, maka semakin tinggi juga skor konsumsi gizi di wilayah tersebut.
                - Perbedaan tinggi batang pada setiap tahun menunjukkan adanya perubahan skor konsumsi gizi dari waktu ke waktu.
                - Grafik ini membantu melihat wilayah mana yang memiliki skor konsumsi gizi lebih tinggi maupun lebih rendah dibanding wilayah lainnya.
                """)


# --- HALAMAN PREDIKSI ---
elif pilihan == "Prediksi":
    st.title("Modul Prediksi XGBoost")

    if not st.session_state.get('data_valid', False):
        st.error(
            "Akses diblokir. Modul Prediksi tidak dapat dijalankan karena dataset belum dipilih atau file yang diunggah tidak valid."
        )
        st.warning(
            "Silakan kembali ke menu Dataset dan pilih sumber data atau unggah file dengan kategori yang benar."
        )

    else:
        with st.expander("Penjelasan Istilah"):
            st.write("""
            1. Skor Aktual adalah nilai konsumsi gizi nyata yang tercatat pada data.
            2. Prediksi XGBoost adalah estimasi nilai konsumsi gizi di masa depan yang dihitung oleh model.
            3. MAE dan RMSE menunjukkan tingkat kesalahan model. Semakin kecil nilainya, semakin baik hasil prediksi.
            4. R-Squared menunjukkan kemampuan model dalam menjelaskan variasi data.
            """)

        df_prediksi = st.session_state.get('df_aktif', df_dummy)

        st.markdown("### Input Parameter Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            pilih_kabkota = st.selectbox(
                "Pilih Kabupaten/Kota:",
                sorted(df_prediksi['Kabupaten/Kota'].unique().tolist())
            )

        with col2:
            pilih_tahun = st.selectbox(
                "Pilih Tahun Basis:",
                sorted(df_prediksi['Tahun'].unique().tolist(), reverse=True)
            )

        if st.button("Proses Prediksi", type="primary", use_container_width=True):
            if xgb_model is None or fitur_model is None:
                st.error(
                    "Model gagal dimuat. Pastikan file model_xgboost_skripsi.pkl dan file pickle lainnya berada di folder yang sama."
                )

            else:
                with st.spinner('Menjalankan algoritma XGBoost...'):
                    data_row = df_prediksi[
                        (df_prediksi['Kabupaten/Kota'] == pilih_kabkota) &
                        (df_prediksi['Tahun'] == pilih_tahun)
                    ]

                    if not data_row.empty:
                        skor_sekarang = float(data_row['Skor Konsumsi Gizi'].values[0])
                        miskin_sekarang = float(data_row['Persentase Miskin (%)'].values[0])
                        provinsi = data_row['Provinsi'].values[0] if 'Provinsi' in data_row.columns else "Banten"

                        input_data = pd.DataFrame(0.0, index=[0], columns=fitur_model)

                        nama_kolom_miskin = (
                            'Penduduk_Miskin (%)'
                            if 'Penduduk_Miskin (%)' in fitur_model
                            else 'Persentase Miskin (%)'
                        )

                        input_data[nama_kolom_miskin] = miskin_sekarang

                        col_prov_name = f"Provinsi_{provinsi}"
                        if col_prov_name in input_data.columns:
                            input_data[col_prov_name] = 1.0

                        try:
                            prediksi = xgb_model.predict(input_data[fitur_model])[0]
                            skor_tahun_depan = round(float(prediksi), 2)
                            trend = round(skor_tahun_depan - skor_sekarang, 2)

                            st.divider()
                            st.markdown(f"### Hasil Prediksi Wilayah {pilih_kabkota}")

                            out1, out2, out3 = st.columns(3)

                            out1.metric(
                                f"Skor Aktual ({pilih_tahun})",
                                f"{skor_sekarang}"
                            )

                            out2.metric(
                                "Penduduk Miskin",
                                f"{miskin_sekarang}%"
                            )

                            out3.metric(
                                f"Prediksi XGBoost ({pilih_tahun + 1})",
                                f"{skor_tahun_depan}",
                                delta=f"{trend}"
                            )

                            st.markdown("#### Catatan Analisis")

                            if skor_sekarang >= 80:
                                st.success(
                                    f"Kondisi gizi di {pilih_kabkota} terpantau baik dan memenuhi standar kecukupan."
                                )
                            elif skor_sekarang >= 70:
                                st.warning(
                                    f"Kondisi gizi di {pilih_kabkota} masuk kategori sedang. Perlu pemantauan agar tidak menurun."
                                )
                            else:
                                st.error(
                                    f"Kondisi gizi di {pilih_kabkota} tergolong rendah. Dibutuhkan intervensi program pangan segera."
                                )

                            st.success(f"""
                            Berdasarkan data, tingkat kemiskinan di {pilih_kabkota} dapat memengaruhi daya beli masyarakat terhadap pangan bergizi.
                            Masyarakat dengan kondisi ekonomi rendah cenderung memilih pangan yang lebih terjangkau,
                            sehingga skor konsumsi gizi dapat sulit meningkat apabila angka kemiskinan belum ditekan.
                            """)

                            st.divider()
                            st.subheader("Evaluasi Performa Model")

                            st.warning(
                                "Perhatian: Nilai metrik MAE, RMSE, dan R-Squared di bawah ini merupakan hasil pengujian model terhadap keseluruhan dataset, bukan evaluasi khusus untuk kabupaten/kota yang sedang dipilih."
                            )

                            tab1, tab2, tab3 = st.tabs([
                                "Skenario Split Data 70:30",
                                "Skenario Split Data 80:20",
                                "Perbandingan Split Data"
                            ])

                            with tab1:
                                c1, c2, c3 = st.columns(3)

                                c1.metric(
                                    "Mean Absolute Error (MAE)",
                                    f"{evaluasi_model['70:30']['mae']:.2f}"
                                )

                                c2.metric(
                                    "Root Mean Squared Error (RMSE)",
                                    f"{evaluasi_model['70:30']['rmse']:.2f}"
                                )

                                c3.metric(
                                    "R-Squared",
                                    f"{evaluasi_model['70:30']['r2']:.2f}"
                                )

                                fig70, ax70 = plt.subplots()
                                ax70.scatter(
                                    visualisasi_model['70:30']['y_test'],
                                    visualisasi_model['70:30']['preds'],
                                    alpha=0.7,
                                    color='#4F8A4B'
                                )

                                ax70.plot(
                                    [
                                        min(visualisasi_model['70:30']['y_test']),
                                        max(visualisasi_model['70:30']['y_test'])
                                    ],
                                    [
                                        min(visualisasi_model['70:30']['y_test']),
                                        max(visualisasi_model['70:30']['y_test'])
                                    ],
                                    color='#C7A72C',
                                    linestyle='--'
                                )

                                ax70.set_xlabel("Nilai Aktual")
                                ax70.set_ylabel("Nilai Hasil Prediksi")
                                ax70.set_title("Sebaran Aktual vs Prediksi 70:30")
                                st.pyplot(fig70)

                                with st.expander("Cara Membaca Grafik 70:30"):
                                    st.write("""
                                    Garis putus-putus menunjukkan garis ideal. Jika titik data berada dekat dengan garis tersebut,
                                    maka hasil prediksi model semakin mendekati nilai aktual. Semakin jauh titik dari garis,
                                    semakin besar perbedaan antara nilai prediksi dan data aktual.
                                    """)

                            with tab2:
                                c1, c2, c3 = st.columns(3)

                                c1.metric(
                                    "Mean Absolute Error (MAE)",
                                    f"{evaluasi_model['80:20']['mae']:.2f}"
                                )

                                c2.metric(
                                    "Root Mean Squared Error (RMSE)",
                                    f"{evaluasi_model['80:20']['rmse']:.2f}"
                                )

                                c3.metric(
                                    "R-Squared",
                                    f"{evaluasi_model['80:20']['r2']:.2f}"
                                )

                                fig80, ax80 = plt.subplots()
                                ax80.scatter(
                                    visualisasi_model['80:20']['y_test'],
                                    visualisasi_model['80:20']['preds'],
                                    alpha=0.7,
                                    color='#6FAF65'
                                )

                                ax80.plot(
                                    [
                                        min(visualisasi_model['80:20']['y_test']),
                                        max(visualisasi_model['80:20']['y_test'])
                                    ],
                                    [
                                        min(visualisasi_model['80:20']['y_test']),
                                        max(visualisasi_model['80:20']['y_test'])
                                    ],
                                    color='#C7A72C',
                                    linestyle='--'
                                )

                                ax80.set_xlabel("Nilai Aktual")
                                ax80.set_ylabel("Nilai Hasil Prediksi")
                                ax80.set_title("Sebaran Aktual vs Prediksi 80:20")
                                st.pyplot(fig80)

                                with st.expander("Cara Membaca Grafik 80:20"):
                                    st.write("""
                                    Grafik ini menunjukkan perbandingan nilai aktual dan nilai prediksi pada skenario pembagian data 80:20.
                                    Titik yang semakin dekat dengan garis ideal menunjukkan hasil prediksi yang lebih baik.
                                    """)

                            with tab3:
                                st.subheader("Hasil Perbandingan Skenario 70:30 dan 80:20")
                                st.write(
                                    "Berikut adalah perbandingan performa model antara dua skenario split data."
                                )

                                df_compare = pd.DataFrame({
                                    "Skenario": ["70:30", "80:20"],
                                    "MAE": [
                                        evaluasi_model['70:30']['mae'],
                                        evaluasi_model['80:20']['mae']
                                    ],
                                    "RMSE": [
                                        evaluasi_model['70:30']['rmse'],
                                        evaluasi_model['80:20']['rmse']
                                    ],
                                    "R-Squared": [
                                        evaluasi_model['70:30']['r2'],
                                        evaluasi_model['80:20']['r2']
                                    ]
                                })

                                st.table(df_compare)

                                mae_70 = evaluasi_model['70:30']['mae']
                                mae_80 = evaluasi_model['80:20']['mae']

                                r2_70 = evaluasi_model['70:30']['r2']
                                r2_80 = evaluasi_model['80:20']['r2']

                                best_scenario = "70:30" if r2_70 > r2_80 else "80:20"
                                selisih_mae = abs(mae_70 - mae_80)

                                st.markdown("### Analisis Hasil Perbandingan")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.info("""
                                    MAE dan RMSE digunakan untuk melihat tingkat kesalahan prediksi model.
                                    Semakin kecil nilainya, maka semakin rendah tingkat kesalahan prediksi.
                                    R-Squared digunakan untuk melihat kemampuan model dalam menjelaskan variasi data.
                                    """)

                                with col2:
                                    st.success(f"""
                                    Berdasarkan hasil pengujian, skenario {best_scenario} menunjukkan performa yang lebih baik
                                    karena memiliki nilai R-Squared yang lebih tinggi dibandingkan skenario lainnya.
                                    """)

                                st.write(f"""
                                Catatan akademik: Perbedaan MAE sebesar {selisih_mae:.4f} menunjukkan bahwa pembagian data
                                dapat memengaruhi hasil evaluasi model. Namun, hasil evaluasi tetap perlu dilihat secara menyeluruh,
                                terutama dari nilai error dan R-Squared.
                                """)

                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat memproses data prediksi: {e}")

# --- HALAMAN ABOUT ---
elif pilihan == "About":
    st.title("Tentang Sistem dan Peneliti")
    st.write(
        "Halaman ini memuat informasi mengenai latar belakang penelitian, tujuan sistem, serta spesifikasi teknis yang digunakan."
    )

    st.divider()

    col_latar, col_tujuan = st.columns(2)

    with col_latar:
        st.markdown("### Latar Belakang Perancangan")
        st.info("""
        Kualitas asupan gizi masyarakat sangat dipengaruhi oleh kondisi sosial ekonomi di suatu wilayah.
        Tingkat kemiskinan sering menjadi salah satu faktor yang membatasi kemampuan rumah tangga dalam
        mengakses pangan bergizi. Oleh karena itu, penelitian ini mencoba memodelkan hubungan antara angka
        kemiskinan dan capaian skor konsumsi gizi menggunakan data dari BPS dan Badan Pangan Nasional.
        """)

    with col_tujuan:
        st.markdown("### Tujuan Sistem")
        st.success("""
        Sistem ini dikembangkan untuk menerjemahkan cara kerja model machine learning ke dalam dashboard
        yang interaktif dan mudah digunakan. Melalui aplikasi ini, pengguna dapat memantau tren data secara visual
        sekaligus melakukan simulasi prediksi skor konsumsi gizi di masa mendatang.
        """)

    st.divider()

    col_prof, col_info = st.columns([1, 1.2])

    with col_prof:
        st.markdown("### Profil Peneliti")
        st.markdown("""
        | Keterangan | Informasi |
        | :--- | :--- |
        | **Nama** | Naswa Azahra |
        | **NIM** | 535220252 |
        | **Program Studi** | Teknik Informatika |
        | **Instansi** | Universitas Tarumanagara |
        | **Dosen Pembimbing Utama** | Desi Arisandi, S.Kom., M.T.I. |
        | **Dosen Pembimbing Pendamping** | Manatap Dolok Lauro Sitorus, S.Kom., M.M.S.I. |
        """)

    with col_info:
        st.markdown("### Spesifikasi Teknis")
        with st.expander("Lihat Detail Teknologi", expanded=True):
            st.write("**Algoritma Utama:**")
            st.code("Extreme Gradient Boosting (XGBoost)", language="text")

            st.write("**Metode Evaluasi:**")
            st.markdown("""
            - Split data: skenario 70:30 dan 80:20.
            - Metrik evaluasi: MAE, RMSE, dan R-Squared.
            """)

            st.write("**Pra-pemrosesan:**")
            st.write(
                "Encoding wilayah menggunakan One-Hot Encoding untuk menangani fitur kategorikal."
            )

    st.divider()

    st.markdown(
        "<p class='footer-text'>2026 - Sistem Prediksi Gizi XGBoost | Skripsi Teknik Informatika UNTAR</p>",
        unsafe_allow_html=True
    )