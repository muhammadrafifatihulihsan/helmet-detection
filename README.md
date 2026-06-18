# 🪖 Helmet Detection (Deteksi Pengendara Helm & Non-Helm)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://ultralytics.com/)

Aplikasi berbasis web ini menggunakan model **YOLO (You Only Look Once)** untuk mendeteksi secara otomatis apakah seorang pengendara menggunakan helm atau tidak dari sebuah gambar. Proyek ini mencakup mulai dari penyiapan data, analisis, pelatihan model, hingga *deployment* menggunakan **Streamlit**.

## ✨ Fitur Utama
- **Deteksi Objek Real-time**: Mengidentifikasi pengendara dengan helm (`Dengan Helm`) dan tanpa helm (`Tanpa Helm`).
- **Confidence Threshold yang Dapat Diatur**: Pengguna dapat menyesuaikan seberapa ketat model mendeteksi objek.
- **Antarmuka Pengguna yang Intuitif**: Dibangun menggunakan Streamlit sehingga interaktif, mudah diakses dan digunakan.
- **Statistik Deteksi**: Menampilkan total objek terdeteksi beserta nilai rata-rata akurasi (*confidence*).

## 📊 Fokus Analisis & Hasil Eksplorasi
Proyek ini tidak hanya berfokus pada hasil akhir berupa aplikasi *end-to-end*, tetapi juga mencakup tahapan analisis data yang mendalam guna membangun model Computer Vision yang handal. Berikut adalah fokus analisis dalam pengembangan model:

1. **Data Preparation & Eksplorasi (`01_data_preparation.ipynb`, `02_data_exploration.ipynb`)**:
   - **Sumber Dataset**: Dataset yang digunakan pada proyek ini dapat diunduh melalui Kaggle: [Helmet Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection).
   - Memeriksa kelengkapan *bounding box* untuk setiap anotasi gambar.
   - Menganalisis distribusi kelas antara pengendara dengan helm dan tanpa helm untuk mengatasi ketidakseimbangan data (*data imbalance*).
   - Meninjau variasi rasio aspek (*aspect ratio*) dan dimensi gambar agar pembuatan *anchor box* pada YOLO dapat dioptimalkan.
2. **Pelatihan Model (`03_training.ipynb`)**:
   - Melatih arsitektur YOLO menggunakan dataset yang sudah dipersiapkan dan dianotasi.
   - Memonitor metrik *loss* selama iterasi (seperti *box loss*, *cls loss*, dan *dfl loss*) untuk mencegah *overfitting* maupun *underfitting*.
3. **Evaluasi Metrik (`04_evaluate.ipynb`)**:
   - Menganalisis kurva evaluasi seperti **Precision**, **Recall**, dan **mAP (Mean Average Precision)**.
   - Mengkaji capaian mAP50 dan mAP50-95 guna memastikan konsistensi dan akurasi model dalam membedakan antara pengendara yang patuh (berhelm) dan yang melanggar di berbagai kondisi gambar.

## 📂 Struktur Direktori
```text
helmet-detection/
├── app/
│   ├── app.py               # Main file untuk menjalankan aplikasi Streamlit
│   ├── requirements.txt     # Dependensi khusus untuk aplikasi web
│   └── models/
│       └── best.pt          # Model YOLO yang sudah dilatih (Pre-trained model)
├── src/
│   ├── dataset/             # Berisi dataset gambar dan label untuk training
│   ├── img/                 # Gambar referensi dan output visualisasi
│   ├── models/              # Penyimpanan model hasil training/eksperimen
│   └── notebooks/           # Jupyter notebooks untuk analisis, training, dan evaluasi
│       ├── 00_setup_gpu.ipynb
│       ├── 01_data_preparation.ipynb
│       ├── 02_data_exploration.ipynb
│       ├── 03_training.ipynb
│       └── 04_evaluate.ipynb
├── .gitignore               # Konfigurasi file yang diabaikan oleh Git
├── README.md                # Dokumentasi utama proyek (File ini)
└── requirements.txt         # Dependensi keseluruhan proyek untuk tahap Development
```

## 🚀 Prasyarat & Instalasi

Pastikan Anda memiliki **Python 3.10+** yang terinstal di sistem Anda.

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/username/helmet-detection.git
   cd helmet-detection
   ```

2. **Buat Virtual Environment (Sangat Disarankan):**
   ```bash
   python -m venv venv
   
   # Untuk Windows:
   venv\Scripts\activate
   
   # Untuk Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependensi Aplikasi:**
   Untuk menjalankan aplikasinya, navigasikan ke folder `app` dan install dependensi:
   ```bash
   cd app
   pip install -r requirements.txt
   ```
   *(Opsional: Anda juga dapat menjalankan `pip install -r requirements.txt` di root folder jika ingin menjalankan dan mengeksekusi Jupyter Notebook untuk bereksperimen).*

## 🖥️ Cara Menjalankan Aplikasi

1. Pastikan *virtual environment* Anda sudah aktif.
2. Pastikan posisi terminal saat ini berada pada direktori `app` (`cd app`).
3. Jalankan perintah server Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka tautan lokal yang diberikan di terminal (secara *default* adalah `http://localhost:8501`) pada *web browser* Anda.

## 📸 Penggunaan
1. Buka aplikasi web di browser.
2. Unggah gambar (mendukung format `.jpg`, `.jpeg`, `.png`, dll.) pada kolom *file uploader* yang disediakan.
3. Atur *slider* **Confidence Threshold** untuk menyesuaikan tingkat kepekaan model (nilai standar: 0.50). Menurunkan threshold akan mendeteksi lebih banyak objek, sementara menaikkannya akan memperketat deteksi.
4. Hasil deteksi berupa *bounding box* berserta jumlah kalkulasi orang berhelm dan tanpa helm akan muncul secara otomatis.

<!-- ## 📄 Lisensi
Proyek ini bersifat *Open Source*. Silakan melakukan *fork* dan memodifikasi aplikasi maupun pipeline *training* sesuai kebutuhan eksplorasi Anda. -->
# helmet-detection
