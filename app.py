import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

# --- 1. SETUP HALAMAN WEB (Harus paling atas) ---
st.set_page_config(
    page_title="Analisis Kecanduan HP", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Utama dengan Gaya
st.title("📱 An Analysis of Teen Addiction on Smartphone")
st.markdown("""
**Analisis Statistik Faktor Penyebab Kecanduan Smartphone pada Remaja** *Project Data Analytics - Python & Streamlit*
---
""")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file ada di folder yang sama
    df = pd.read_csv('teen_phone_addiction_dataset.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ File dataset tidak ditemukan! Mohon pastikan file csv ada di folder yang sama.")
    st.stop()

# --- 3. SIDEBAR (MENU KIRI) ---
with st.sidebar:
    st.header("⚙️ Kontrol Dashboard")
    st.info("Gunakan menu ini untuk memfilter tampilan.")
    tampilkan_raw = st.checkbox("🔍 Tampilkan Data Mentah")
    
    st.divider()
    st.write("Developed by Group 3 Data Analaytics:")
    st.caption("Anastasya Putri Wahyudi Uloli")
    st.caption("Raffi Aria Habibie")
    st.caption("Muhammad Abel Maulana")
    st.caption("Shira Audrey Kanira")

# --- 4. PREPROCESSING (FILTERING) ---
col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    st.subheader("1. Preprocessing & Data Cleaning")
    st.markdown("Kami melakukan pembersihan data outlier menggunakan metode **Confidence Interval (95%)**.")

    # Hitung Batas
    lcb = df['Addiction_Level'].mean() - 1.96 * df['Addiction_Level'].std()
    ucb = df['Addiction_Level'].mean() + 1.96 * df['Addiction_Level'].std()

    # Filter Data
    data_clean = df[(df['Addiction_Level'] <= ucb) & (df['Addiction_Level'] >= lcb)].copy()

    # Tampilkan Metrics dalam kolom kecil
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sampel Awal", f"{len(df)} Remaja")
    m2.metric("Data Bersih (Wajar)", f"{len(data_clean)} Remaja")
    m3.metric("Data Dibuang (Outlier)", f"{len(df) - len(data_clean)}")
    
    st.caption(f"ℹ️ Batas skor kecanduan yang dianggap wajar: **{lcb:.2f}** s.d **{ucb:.2f}**")

with col_main2:
    if tampilkan_raw:
        st.write("**Preview Data:**")
        st.dataframe(data_clean.head(10), height=250)
    else:
        st.write("*(Centang 'Tampilkan Data Mentah' di sidebar untuk melihat tabel)*")

st.divider()

# --- 5. EXPLORATORY DATA ANALYSIS (EDA) ---
st.subheader("2. Exploratory Data Analysis (EDA)")

tab1, tab2, tab3 = st.tabs(["📊 Distribusi Data", "🧩 Analisis Kategori", "🔗 Korelasi Linear"])

with tab1:
    st.markdown("#### Bagaimana sebaran tingkat kecanduan remaja?")
    
    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Style Plot
        ax.grid(True, linestyle='--', alpha=0.3)
        
        kde = stats.gaussian_kde(data_clean['Addiction_Level'])
        pos = np.linspace(data_clean['Addiction_Level'].min()-1, data_clean['Addiction_Level'].max()+1, 1000)
        
        lower_conf = data_clean['Addiction_Level'].quantile(0.05)
        upper_conf = data_clean['Addiction_Level'].quantile(0.95)
        shade = np.linspace(lower_conf, upper_conf, 300)
        
        ax.plot(pos, kde(pos), color='#2E86C1', linewidth=2.5)
        ax.fill_between(shade, kde(shade), alpha=0.3, color='#2E86C1', label='95% Confidence Interval')
        
        ax.set_title('Density Plot: Tingkat Kecanduan (Smoothed)', fontsize=12)
        ax.set_xlabel('Skor Kecanduan')
        ax.set_ylabel('Density')
        ax.legend()
        st.pyplot(fig)
    
    with col_g2:
        st.info("""
        **Analisis:**
        Grafik ini menunjukkan di angka berapa skor kecanduan paling banyak berkumpul.
        Area biru muda menunjukkan di mana 95% remaja berada.
        """)

with tab2:
    st.markdown("#### Apakah durasi penggunaan mempengaruhi kecanduan?")
    
    # Feature Engineering
    data_clean['Usage_Category'] = pd.cut(
        data_clean['Daily_Usage_Hours'], 
        bins=[0, 4, 7, 24], 
        labels=['Low (<4h)', 'Medium (4-7h)', 'High (>7h)']
    )
    
    c1, c2 = st.columns(2)
    with c1:
        # Boxplot dengan perbaikan sintaks (Fix Warning)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=data_clean, x='Usage_Category', y='Addiction_Level', 
                    hue='Usage_Category', legend=False, palette='viridis', ax=ax2)
        ax2.set_title("Distribusi Kecanduan per Kategori Pengguna")
        ax2.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig2)
        
    with c2:
        st.write("**Statistik Deskriptif per Kategori:**")
        hasil_group = data_clean.groupby('Usage_Category', observed=False).agg(
            {'Addiction_Level': ['mean', 'min', 'max', 'count']}
        ).round(2)
        st.dataframe(hasil_group, use_container_width=True)
        st.success("Terlihat jelas bahwa kategori **High** memiliki rata-rata skor kecanduan tertinggi.")

with tab3:
    st.markdown("#### Hubungan antara Durasi (Jam) dan Skor Kecanduan")
    col_sc1, col_sc2 = st.columns([3,1])
    with col_sc1:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.regplot(data=data_clean, x='Daily_Usage_Hours', y='Addiction_Level', 
                    line_kws={'color':'red', 'linewidth': 2}, 
                    scatter_kws={'alpha':0.3, 'color': 'teal'}, ax=ax3)
        ax3.set_title("Scatter Plot + Regression Line")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
    with col_sc2:
        st.write("Garis merah menunjukkan tren: Semakin ke kanan (makin lama main), garis makin naik (makin candu).")

st.divider()

# --- 6. REGRESSION ANALYSIS ---
st.subheader("3. Regression Analysis (Model Statistik)")
st.markdown("Kami menggunakan **Multiple Linear Regression** untuk mengukur seberapa besar pengaruh variabel bebas terhadap tingkat kecanduan.")

# Model
X = data_clean[['Daily_Usage_Hours', 'Phone_Checks_Per_Day', 'Time_on_Social_Media', 'Age']]
Y = data_clean['Addiction_Level']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()

# Tampilan Hasil yang Rapi
col_res1, col_res2 = st.columns(2)

with col_res1:
    st.markdown("##### 🎯 Kekuatan Model (R-Squared)")
    r_sq = model.rsquared
    st.metric("R-Squared", f"{r_sq*100:.2f}%", help="Persentase variasi kecanduan yang bisa dijelaskan oleh model.")
    st.info(f"""
    **Interpretasi:**
    Variabel yang kita analisis (Durasi, Sosmed, Cek HP, Umur) mampu menjelaskan **{r_sq*100:.1f}%** penyebab kecanduan. 
    Sisanya dipengaruhi faktor lain di luar data ini.
    """)

with col_res2:
    st.markdown("##### 📊 Signifikansi Variabel")
    # Mengambil tabel summary dan merapikannya
    summary_df = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    
    # Highlight P-Value
    def highlight_significance(val):
        color = '#d4edda' if val < 0.05 else '#f8d7da' # Hijau jika signifikan, Merah jika tidak
        return f'background-color: {color}'

    st.dataframe(summary_df[['coef', 'P>|t|']].style.applymap(highlight_significance, subset=['P>|t|']).format("{:.4f}"))
    st.caption("*Hijau = Berpengaruh Signifikan (< 0.05), Merah = Tidak Berpengaruh*")

# --- 7. SIMULATOR (BONUS ROCKET) ---
st.divider()
st.subheader("🚀 Simulator Prediksi Kecanduan")
st.markdown("Cobalah ubah slider di bawah ini untuk memprediksi skor kecanduan berdasarkan model yang telah dibuat.")

# Input User
col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
with col_sim1:
    input_usage = st.number_input("Durasi Harian (Jam)", 0.0, 24.0, 5.0)
with col_sim2:
    input_sosmed = st.number_input("Durasi Sosmed (Jam)", 0.0, 24.0, 3.0)
with col_sim3:
    input_checks = st.number_input("Cek HP per Hari", 0, 500, 50)
with col_sim4:
    input_age = st.number_input("Umur", 10, 25, 18)

# Tombol Prediksi
if st.button("Hitung Prediksi Skor"):
    # Rumus Manual dari Koefisien Regresi: Y = C + b1X1 + b2X2 ...
    params = model.params
    prediksi = (params['const'] + 
                params['Daily_Usage_Hours'] * input_usage + 
                params['Phone_Checks_Per_Day'] * input_checks + 
                params['Time_on_Social_Media'] * input_sosmed + 
                params['Age'] * input_age)
    
    st.success(f"### Prediksi Tingkat Kecanduan: {prediksi:.2f} / 10")
    
    if prediksi > 8:
        st.warning("⚠️ Hati-hati! Skor ini tergolong SANGAT TINGGI.")
    elif prediksi > 5:
        st.info("ℹ️ Skor Menengah.")
    else:
        st.success("✅ Skor Aman.")