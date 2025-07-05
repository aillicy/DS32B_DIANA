import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
import plotly.express as px 
import plotly.graph_objects as go 
from datetime import datetime, timedelta

# --- Konfigurasi Halaman (Harus di awal skrip) ---
st.set_page_config(
    page_title="Dashboard Analisis Penjualan",
    page_icon="📊",
    layout="wide", # Layout 'wide' membuat konten lebih lebar
    initial_sidebar_state="expanded" # Sidebar langsung terbuka (visible) saat user pertama kali mengakses app
)

# --- Fungsi untuk Memuat Data Dummy Penjualan (Menggunakan Cache untuk Performa) ---
@st.cache_data
def load_data():
    return pd.read_csv("data/retail_store.csv")

# Load data penjualan
df_sales = load_data()
df_sales['Tanggal_Pesanan'] = pd.to_datetime(df_sales['Tanggal_Pesanan']) # Mengubah ke datetime

# --- Fungsi untuk Melatih Model Regresi (Menggunakan Cache) ---
@st.cache_resource
def load_model():
    with open("model/model_sales.pkl", "rb") as f:
        sales_prediction_model, model_features, base_month_ordinal = pickle.load(f)
    return sales_prediction_model, model_features, base_month_ordinal

# Load model
sales_prediction_model, model_features, base_month_ordinal = load_model()


# --- Judul Dashboard ---
st.title("📈 Dashboard Analisis Penjualan Toko Online 🛍️")
st.markdown("Dashboard interaktif ini menyediakan gambaran umum performa penjualan, tren, dan distribusi berdasarkan berbagai dimensi, **serta fitur prediksi sederhana**.")

st.markdown("---") # Garis pembatas

# --- Sidebar untuk Filter Global dan Navigasi ---
st.sidebar.header("⚙️ Pengaturan & Navigasi")

pilihan_halaman = st.sidebar.radio(
    "Pilih Halaman:",
    ("Overview Dashboard", "Prediksi Penjualan")
)

# Filter Global (hanya muncul jika di halaman Overview Dashboard)
if pilihan_halaman == "Overview Dashboard":
    st.sidebar.markdown("### Filter Data Dashboard")
    # Filter berdasarkan tanggal (rentang)
    min_date = df_sales['Tanggal_Pesanan'].min().date()
    max_date = df_sales['Tanggal_Pesanan'].max().date()

    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Pastikan date_range memiliki 2 elemen
    if len(date_range) == 2:
        start_date_filter = pd.to_datetime(date_range[0])
        end_date_filter = pd.to_datetime(date_range[1])
        filtered_df = df_sales[(df_sales['Tanggal_Pesanan'] >= start_date_filter) & 
                               (df_sales['Tanggal_Pesanan'] <= end_date_filter)]
    else:
        # Handle case where only one date is selected (e.g., initial state)
        filtered_df = df_sales


    # Filter berdasarkan Wilayah
    selected_regions = st.sidebar.multiselect(
        "Pilih Wilayah:",
        options=df_sales['Wilayah'].unique().tolist(),
        default=df_sales['Wilayah'].unique().tolist()
    )
    filtered_df = filtered_df[filtered_df['Wilayah'].isin(selected_regions)]

    # Filter berdasarkan Kategori Produk
    selected_categories = st.sidebar.multiselect(
        "Pilih Kategori Produk:",
        options=df_sales['Kategori'].unique().tolist(),
        default=df_sales['Kategori'].unique().tolist()
    )
    filtered_df = filtered_df[filtered_df['Kategori'].isin(selected_categories)]
else: # If on Prediction Page, use full df for the model
    filtered_df = df_sales.copy()


# Jika tidak ada data setelah filter (hanya relevan di Overview)
if pilihan_halaman == "Overview Dashboard" and filtered_df.empty:
    st.warning("Tidak ada data yang tersedia berdasarkan filter yang Anda pilih. Silakan sesuaikan filter.")
    st.stop() # Menghentikan eksekusi skrip jika tidak ada data


# --- Konten Halaman Utama Berdasarkan Pilihan Halaman ---

if pilihan_halaman == "Overview Dashboard":
    # --- Metrik Utama (Menggunakan st.columns dan st.metric) ---
    st.subheader("📊 Ringkasan Performa Penjualan")

    col1, col2, col3, col4 = st.columns(4)

    total_sales = filtered_df['Total_Penjualan'].sum()
    total_orders = filtered_df['OrderID'].nunique()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    total_products_sold = filtered_df['Jumlah'].sum()

    with col1:
        st.metric(label="Total Penjualan", value=f"Rp {total_sales:,.2f}")
    with col2:
        st.metric(label="Jumlah Pesanan", value=f"{total_orders:,}")
    with col3:
        st.metric(label="Rata-rata Nilai Pesanan", value=f"Rp {avg_order_value:,.2f}")
    with col4:
        st.metric(label="Jumlah Produk Terjual", value=f"{total_products_sold:,}")

    st.markdown("---")

    # --- Tren Penjualan Bulanan (Line Chart) ---
    st.subheader("📈 Tren Penjualan Bulanan")
    sales_by_month = filtered_df.groupby('Bulan')['Total_Penjualan'].sum().reset_index()
    # Pastikan urutan bulan benar
    sales_by_month['Bulan'] = pd.to_datetime(sales_by_month['Bulan']).dt.to_period('M')
    sales_by_month = sales_by_month.sort_values('Bulan')
    sales_by_month['Bulan'] = sales_by_month['Bulan'].astype(str) # Kembali ke string untuk Plotly

    fig_monthly_sales = px.line(
        sales_by_month,             # Data agregat penjualan per bulan (DataFrame)
        x='Bulan',                  # Sumbu X: bulan (format string seperti '2024-01')
        y='Total_Penjualan',        # Sumbu Y: total penjualan
        title='Total Penjualan per Bulan',  # Judul grafik
        markers=True,               # Menampilkan titik pada setiap nilai (marker)
        line_shape="spline",        # Membuat garis melengkung (bukan garis lurus biasa)
        hover_name='Bulan',         # Menampilkan nama bulan saat kursor diarahkan ke titik
        height=400                  # Tinggi grafik dalam piksel
    )

    # Update warna garis menjadi hijau
    fig_monthly_sales.update_traces(line_color='#2ca02c')

    # Menampilkan grafik di halaman Streamlit, lebarnya menyesuaikan container (misalnya kolom atau layar)
    st.plotly_chart(fig_monthly_sales, use_container_width=True)

    st.markdown("---")

    # --- Distribusi Penjualan & Produk Terlaris (2 Kolom) ---
    st.subheader("Top Produk & Distribusi Penjualan")

    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        st.write("#### Top 10 Produk Terlaris (Berdasarkan Total Penjualan)")

        # Agregasi total penjualan per produk, ambil 10 produk dengan penjualan tertinggi
        top_products_sales = filtered_df.groupby('Produk')['Total_Penjualan'].sum().nlargest(10).reset_index()

        # Membuat horizontal bar chart menggunakan Plotly Express
        fig_top_products = px.bar(
            top_products_sales,
            x='Total_Penjualan',     # Panjang batang = total penjualan
            y='Produk',              # Y axis = nama produk
            orientation='h',         # Horizontal bar
            title='Top 10 Produk Berdasarkan Total Penjualan',
            color='Total_Penjualan', # Warna batang berdasarkan nilai
            color_continuous_scale=px.colors.sequential.Plasma[::-1],  # Gradasi warna dari terang ke gelap
            height=400
        )

        # Sortir kategori (produk) berdasarkan nilai total penjualan (ascending)
        fig_top_products.update_layout(yaxis={'categoryorder':'total ascending'})

        # Tampilkan grafik di Streamlit
        st.plotly_chart(fig_top_products, use_container_width=True)


    with col_vis2:
        st.write("#### Distribusi Penjualan per Kategori")

        # Agregasi total penjualan berdasarkan kategori
        sales_by_category = filtered_df.groupby('Kategori')['Total_Penjualan'].sum().reset_index()

        # Buat pie chart (donut style) berdasarkan proporsi penjualan tiap kategori
        fig_category_pie = px.pie(
            sales_by_category,
            values='Total_Penjualan',    # Nilai yang diplot (besarannya)
            names='Kategori',            # Label di pie chart
            title='Proporsi Penjualan per Kategori',
            hole=0.3,                    # Membuat pie menjadi donut chart (ada lubangnya)
            color_discrete_sequence=px.colors.qualitative.Set2  # Skema warna yang friendly
        )

        # Tampilkan chart di Streamlit
        st.plotly_chart(fig_category_pie, use_container_width=True)


    st.markdown("---")

    # --- Penjualan Berdasarkan Metode Pembayaran & Wilayah (Menggunakan Tabs) ---
    st.subheader("Performa Penjualan Lebih Detail")

    # Buat dua tab: Tab1 untuk metode pembayaran, Tab2 untuk wilayah
    tab1, tab2 = st.tabs(["Metode Pembayaran", "Penjualan per Wilayah"])

    # --- Tab 1: Visualisasi Penjualan per Metode Pembayaran ---
    with tab1:
        st.write("#### Penjualan Berdasarkan Metode Pembayaran")

        # Hitung total penjualan per metode pembayaran
        sales_by_payment = (
            filtered_df.groupby('Metode_Pembayaran')['Total_Penjualan']
            .sum()
            .reset_index()
        )

        # Buat bar chart berdasarkan metode pembayaran
        fig_payment = px.bar(
            sales_by_payment,
            x='Metode_Pembayaran',
            y='Total_Penjualan',
            title='Total Penjualan per Metode Pembayaran',
            color='Metode_Pembayaran',
            color_discrete_sequence=px.colors.qualitative.Vivid  # Skema warna cerah
        )

        # Tampilkan chart di Streamlit
        st.plotly_chart(fig_payment, use_container_width=True)

    # --- Tab 2: Visualisasi Penjualan per Wilayah ---
    with tab2:
        st.write("#### Penjualan Berdasarkan Wilayah")

        # Hitung total penjualan per wilayah
        sales_by_region = (
            filtered_df.groupby('Wilayah')['Total_Penjualan']
            .sum()
            .reset_index()
        )

        # Buat bar chart berdasarkan wilayah
        fig_region = px.bar(
            sales_by_region,
            x='Wilayah',
            y='Total_Penjualan',
            title='Total Penjualan per Wilayah',
            color='Wilayah',
            color_discrete_sequence=px.colors.qualitative.Safe  # Warna yang lebih lembut
        )

        # Tampilkan chart di Streamlit
        st.plotly_chart(fig_region, use_container_width=True)

    # Tambahkan garis horizontal sebagai pemisah antar bagian
    st.markdown("---")


    # --- Eksplorasi Data Mentah (Expander) ---
    st.subheader("🔬 Eksplorasi Data Mentah")

    # Buat area yang bisa diklik untuk expand/collapse
    with st.expander("Klik untuk melihat detail data transaksi"):

        # Penjelasan singkat
        st.write("Berikut adalah sebagian kecil dari data transaksi yang digunakan untuk dashboard ini.")
        
        # Slider untuk memilih berapa banyak baris data yang ingin ditampilkan
        num_rows_to_display = st.slider(
            "Jumlah Baris Data yang Ditampilkan:",
            min_value=10, # Nilai terkecil yang bisa dipilih pengguna (slider mulai dari angka 10)
            max_value=200, # Nilai terbesar yang bisa dipilih pengguna (slider mentok di 200)
            value=50, # Nilai default saat slider muncul pertama kali
            step=10 # Jarak antar nilai pada slider (misalnya: 10, 20, 30, ..., 200)
        )

        # Tampilkan tabel data sesuai jumlah baris yang dipilih
        st.dataframe(filtered_df.head(num_rows_to_display))
        
        # Tampilkan statistik deskriptif (count, mean, std, min, max, dsb)
        st.write("Statistik Deskriptif:")
        st.dataframe(filtered_df.describe())


elif pilihan_halaman == "Prediksi Penjualan":
    st.header("🔮 Prediksi Penjualan Sederhana")
    st.write("Gunakan bagian ini untuk memprediksi potensi total penjualan berdasarkan parameter yang dipilih. Model ini menggunakan **Regresi Linear** sederhana.")

    st.subheader("Input Parameter Prediksi")

    # Menggunakan kolom untuk input yang rapi
    col_pred_1, col_pred_2 = st.columns(2)

    with col_pred_1:
        # Prediksi untuk tanggal tertentu (kita ubah ke ordinal untuk model)
        target_date = st.date_input("Tanggal Prediksi:", value=datetime.now().date() + timedelta(days=7))
        target_date_ordinal = target_date.toordinal()

        # Input untuk rata-rata Jumlah produk per pesanan
        avg_quantity = st.slider("Rata-rata Jumlah Produk per Pesanan:", min_value=1.0, max_value=5.0, value=df_sales['Jumlah'].mean(), step=0.1)

        # Input untuk rata-rata Harga Satuan
        avg_unit_price = st.slider("Rata-rata Harga Satuan Produk (Rp):", min_value=50.0, max_value=2000.0, value=df_sales['Harga_Satuan'].mean(), step=10.0)

    with col_pred_2:
        # Input untuk rata-rata Diskon
        avg_discount = st.slider("Rata-rata Diskon (%):", min_value=0.0, max_value=0.20, value=df_sales['Diskon'].mean(), step=0.01, format="%.2f")

        # Input untuk Hari dalam Seminggu
        day_of_week = st.selectbox("Hari dalam Seminggu:", options=['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'], index=datetime.now().weekday())
        day_of_week_map = {'Senin':0, 'Selasa':1, 'Rabu':2, 'Kamis':3, 'Jumat':4, 'Sabtu':5, 'Minggu':6}
        day_of_week_encoded = day_of_week_map[day_of_week]

        # Input untuk Jam Pesanan (misal: jam puncak transaksi)
        hour_of_day = st.slider("Jam Puncak Pesanan (0-23):", min_value=0, max_value=23, value=14, step=1)

    st.markdown("---")

    if st.button("Hitung Prediksi Penjualan"):
        # Siapkan input untuk model
        input_for_prediction = pd.DataFrame([[
            target_date_ordinal,
            avg_quantity,
            avg_unit_price,
            avg_discount,
            day_of_week_encoded,
            hour_of_day
        ]], columns=model_features)

        try:
            predicted_sales_value = sales_prediction_model.predict(input_for_prediction)[0]
            
            st.success(f"Berdasarkan parameter yang diberikan, prediksi total penjualan adalah: **Rp {predicted_sales_value:,.2f}**")
            
            st.info("""
            *Catatan: Model ini adalah demonstrasi sederhana menggunakan Regresi Linear dan data dummy. 
            Prediksi aktual membutuhkan model yang lebih kompleks, data historis yang memadai, dan fitur yang lebih relevan.*
            """)

            # Tampilkan fitur yang digunakan untuk prediksi
            st.markdown("---")
            st.subheader("Fitur yang Digunakan untuk Prediksi:")
            st.write(input_for_prediction)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}. Pastikan semua input valid.")


st.markdown("---")
st.caption("Dashboard ini dibuat dengan ❤️ oleh Peserta Bootcamp DSA Dibimbing Batch 32B ")