import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

# --- 1. SETUP HALAMAN WEB ---
st.set_page_config(
    page_title="Analisis Kecanduan HP", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS biar tampilan tabel makin paten & font hitam
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    div[data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Judul Utama
st.title("📱 Analysis of Teen Smartphone Addiction")
st.markdown("""
**Analisis Statistik Faktor Penyebab Kecanduan Smartphone pada Remaja** *Project Data Analytics - Python & Streamlit*
---
""")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('teen_phone_addiction_dataset.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ File dataset tidak ditemukan! Mohon pastikan file csv ada di folder yang sama.")
    st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    st.info("Menu Filter & Opsi Tampilan")
    
    tampilkan_raw = st.checkbox("🔍 Tampilkan Data Mentah")
    
    st.divider()
    st.write("**Developed by Group 3:**")
    st.caption("1. Anastasya Putri Wahyudi Uloli")
    st.caption("2. Raffi Aria Habibie")
    st.caption("3. Muhammad Abel Maulana")
    st.caption("4. Shira Audrey Kanira")

# --- 4. PREPROCESSING ---
# Gunakan container biar rapi
with st.container():
    col_main1, col_main2 = st.columns([2, 1])

    with col_main1:
        st.subheader("1. Preprocessing & Data Cleaning")
        st.write("Metode: **Confidence Interval (95%)** untuk memisahkan data wajar dan outlier.")

        lcb = df['Addiction_Level'].mean() - 1.96 * df['Addiction_Level'].std()
        ucb = df['Addiction_Level'].mean() + 1.96 * df['Addiction_Level'].std()

        data_clean = df[(df['Addiction_Level'] <= ucb) & (df['Addiction_Level'] >= lcb)].copy()

        # --- PERBAIKAN METRIC TERPOTONG ---
        # Label diperpendek biar gak kepotong jadi "Re..."
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Data", f"{len(df)}")
        m2.metric("Data Bersih", f"{len(data_clean)}")
        m3.metric("Outlier Dibuang", f"{len(df) - len(data_clean)}")
        
        st.caption(f"✅ Rentang skor wajar: **{lcb:.2f}** - **{ucb:.2f}**")

    with col_main2:
        if tampilkan_raw:
            st.write("**Preview Data:**")
            st.dataframe(data_clean.head(5), height=200)
        else:
            st.info("💡 Centang 'Tampilkan Data Mentah' di sidebar untuk melihat tabel.")

st.divider()

# --- 5. EXPLORATORY DATA ANALYSIS (EDA) ---
st.subheader("2. Exploratory Data Analysis (EDA)")

tab1, tab2, tab3 = st.tabs(["📊 Distribusi", "🧩 Kategori User", "🔗 Korelasi"])

with tab1:
    st.markdown("##### Sebaran Tingkat Kecanduan")
    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.grid(True, linestyle='--', alpha=0.3)
        
        kde = stats.gaussian_kde(data_clean['Addiction_Level'])
        pos = np.linspace(data_clean['Addiction_Level'].min()-1, data_clean['Addiction_Level'].max()+1, 1000)
        
        lower_conf = data_clean['Addiction_Level'].quantile(0.05)
        upper_conf = data_clean['Addiction_Level'].quantile(0.95)
        shade = np.linspace(lower_conf, upper_conf, 300)
        
        ax.plot(pos, kde(pos), color='#2E86C1', linewidth=2.5)
        ax.fill_between(shade, kde(shade), alpha=0.3, color='#2E86C1', label='95% Area')
        
        ax.set_title('Density Plot: Addiction Level', fontsize=12)
        ax.set_xlabel('Skor Kecanduan')
        ax.legend()
        st.pyplot(fig)
    
    with col_g2:
        st.info("Grafik ini menunjukkan di mana mayoritas skor kecanduan remaja berada (Area Biru).")

with tab2:
    st.markdown("##### Perbandingan Berdasarkan Durasi Main")
    data_clean['Usage_Category'] = pd.cut(
        data_clean['Daily_Usage_Hours'], 
        bins=[0, 4, 7, 24], 
        labels=['Low (<4h)', 'Medium (4-7h)', 'High (>7h)']
    )
    
    c1, c2 = st.columns(2)
    with c1:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=data_clean, x='Usage_Category', y='Addiction_Level', 
                    hue='Usage_Category', legend=False, palette='viridis', ax=ax2)
        ax2.set_title("Distribusi Kecanduan per Kategori")
        ax2.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig2)
        
    with c2:
        st.write("**Statistik Rata-rata:**")
        hasil_group = data_clean.groupby('Usage_Category', observed=False).agg(
            {'Addiction_Level': ['mean', 'max', 'count']}
        ).round(2)
        # Gunakan use_container_width=True biar pas layar
        st.dataframe(hasil_group, use_container_width=True)
        st.success("User **High (>7h)** cenderung memiliki skor kecanduan lebih tinggi.")

with tab3:
    st.markdown("##### Scatter Plot: Durasi vs Kecanduan")
    col_sc1, col_sc2 = st.columns([3,1])
    with col_sc1:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.regplot(data=data_clean, x='Daily_Usage_Hours', y='Addiction_Level', 
                    line_kws={'color':'red', 'linewidth': 2}, 
                    scatter_kws={'alpha':0.3, 'color': 'teal'}, ax=ax3)
        st.pyplot(fig3)
    with col_sc2:
        st.write("Garis merah naik artinya hubungan **Positif**: Makin lama main, makin tinggi skor kecanduan.")

st.divider()

# --- 6. REGRESSION ANALYSIS ---
st.subheader("3. Regression Analysis (Model Statistik)")
st.markdown("Model ini mengukur seberapa besar pengaruh 5 variabel terhadap kecanduan.")

# --- PERBAIKAN R-SQUARED ---
# Tambahkan Opsi: Pakai Data Bersih (Default) atau Semua Data
pilihan_data = st.radio(
    "Pilih Dataset untuk Model:",
    ("Gunakan Data Bersih (Disarankan)", "Gunakan Semua Data (Termasuk Outlier)"),
    horizontal=True
)

# Logic pemilihan data
if pilihan_data == "Gunakan Data Bersih (Disarankan)":
    target_df = data_clean
    st.caption("ℹ️ Menggunakan 2819 data (tanpa outlier). R-Squared mungkin sedikit turun karena variasi ekstrem dibuang.")
else:
    target_df = df
    st.caption("ℹ️ Menggunakan 3000 data asli. R-Squared biasanya lebih tinggi di sini (sekitar 45%).")

# Define Variables
feature_cols = ['Daily_Usage_Hours', 'Phone_Checks_Per_Day', 'Time_on_Social_Media', 'Screen_Time_Before_Bed', 'Age']
X = target_df[feature_cols]
Y = target_df['Addiction_Level']
X = sm.add_constant(X)

# Fit Model
model = sm.OLS(Y, X).fit()

# Tampilan Hasil
col_res1, col_res2 = st.columns(2)

with col_res1:
    st.markdown("##### 🎯 Performansi Model")
    r_sq = model.rsquared
    
    # Tampilkan R-Squared Besar
    st.metric("R-Squared Score", f"{r_sq*100:.2f}%")
    
    st.info(f"""
    **Artinya:** Kelima variabel usage kita berhasil menjelaskan **{r_sq*100:.2f}%** penyebab naik-turunnya skor kecanduan.
    """)

with col_res2:
    st.markdown("##### 📊 Signifikansi & Koefisien")
    
    # Buat dataframe manual biar gak perlu lxml
    summary_df = pd.DataFrame({
        'Variabel': ['Intercept'] + feature_cols,
        'Koefisien': model.params.values,
        'P-Value': model.pvalues.values
    })
    
    # --- PERBAIKAN WARNA TABEL ---
    # Tambahkan 'color: black' supaya tulisan tidak putih/nyaru
    def style_table(val):
        color_bg = '#d4edda' if val < 0.05 else '#f8d7da' # Hijau/Merah background
        return f'background-color: {color_bg}; color: black; font-weight: bold;'

    # Tampilkan tabel
    st.dataframe(
        summary_df.style.applymap(style_table, subset=['P-Value']).format({'Koefisien': '{:.4f}', 'P-Value': '{:.4f}'}),
        use_container_width=True,
        hide_index=True
    )
    st.caption("Hijau = Berpengaruh Signifikan (< 0.05) | Merah = Tidak Berpengaruh")

# --- 7. SIMULATOR (SUDAH DIPERBAIKI: 5 INPUT) ---
st.divider()
st.subheader("🚀 Simulator Prediksi Kecanduan")
st.markdown("Masukkan data kebiasaan pengguna (5 Indikator):")

# Kita bagi jadi 2 baris biar rapi (3 di atas, 2 di bawah)
# BARIS 1: Usage, Sosmed, Checks
row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    input_usage = st.number_input("1. Durasi Harian (Jam)", 0.0, 24.0, 5.0)
with row1_col2:
    input_sosmed = st.number_input("2. Durasi Sosmed (Jam)", 0.0, 24.0, 3.0)
with row1_col3:
    input_checks = st.number_input("3. Cek HP per Hari", 0, 500, 50)

# BARIS 2: Bedtime (YANG HILANG TADI) & Age
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    # INI DIA VARIABEL YANG KAMU CARI
    input_bedtime = st.number_input("4. Main HP Sebelum Tidur (Jam)", 0.0, 10.0, 1.0)
with row2_col2:
    input_age = st.number_input("5. Umur Pengguna", 10, 30, 18)

# Tombol Prediksi
if st.button("Hitung Prediksi Skor"):
    # Ambil nilai koefisien dari model yang sudah dilatih di atas
    params = model.params
    
    # Hitung sumbangsih setiap variabel (Pecah biar gak error syntax)
    # Pastikan nama params['...'] SAMA PERSIS dengan nama kolom di CSV
    skor_dasar   = params['const']
    skor_usage   = params['Daily_Usage_Hours'] * input_usage
    skor_checks  = params['Phone_Checks_Per_Day'] * input_checks
    skor_sosmed  = params['Time_on_Social_Media'] * input_sosmed
    skor_bedtime = params['Screen_Time_Before_Bed'] * input_bedtime  # <-- Sudah masuk rumus
    skor_age     = params['Age'] * input_age
    
    # Jumlahkan semua
    prediksi = skor_dasar + skor_usage + skor_checks + skor_sosmed + skor_bedtime + skor_age
    
    # Tampilkan Hasil
    st.success(f"### Prediksi Tingkat Kecanduan: {prediksi:.2f} / 10")
    
    # Logic Kategori Bahaya
    if prediksi > 8:
        st.error("⚠️ SANGAT TINGGI (Bahaya)")
        st.write("Saran: Kurangi main HP sebelum tidur dan batasi durasi harian.")
    elif prediksi > 5:
        st.warning("⚠️ SEDANG (Waspada)")
    else:
        st.success("✅ RENDAH (Aman)")