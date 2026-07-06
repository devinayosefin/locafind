LOCAFIND - PANDUAN INSTALASI DAN PENGGUNAAN
===========================================

LocaFind adalah aplikasi rekomendasi wisata/kuliner/cafe/pusat perbelanjaan
Surabaya berbasis Flask. User memasukkan hari kunjungan, budget, jam perjalanan,
kategori, dan preferensi tempat. Sistem lalu menyusun itinerary otomatis dengan
memperhatikan relevansi semantik, budget, jam buka, estimasi durasi, dan jarak.


1. LIBRARY YANG PERLU DI-INSTALL
--------------------------------

Minimal gunakan Python 3.10 atau lebih baru.

Library utama:

- flask
  Untuk menjalankan web server dan routing halaman.

- pandas
  Untuk membaca dan memproses dataset.

- openpyxl
  Untuk membaca file Excel Dataset LocaFind.xlsx.

- sentence-transformers
  Untuk membuat dan memakai embedding semantik SBERT/SentenceTransformer.

- scikit-learn
  Untuk cosine similarity dan fallback TF-IDF.

- geopy
  Library pendukung geolocation/distance jika diperlukan.

- requests
  Untuk mendeteksi lokasi user lewat IP.

Cara install semua library:

    pip install -r requirements.txt

Jika belum punya pip terbaru, jalankan:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

Catatan:
Saat pertama kali menjalankan build_model.py, sentence-transformers bisa
men-download model dari Hugging Face. Pastikan internet aktif pada proses build
pertama. Setelah model tersimpan di cache lokal, app.py bisa berjalan offline
untuk memuat model tersebut.


2. STRUKTUR FILE PENTING
------------------------

- app.py
  Entry point web Flask. Mengatur halaman Home dan halaman Hasil.

- build_model.py
  Script preprocessing dataset dan pembuatan embedding semantik.

- engine.py
  Inti sistem rekomendasi dan penyusun itinerary.

- Dataset LocaFind.xlsx
  Dataset tempat wisata/kuliner/cafe/pusat perbelanjaan.

- requirements.txt
  Daftar library Python yang perlu di-install.

- models/
  Folder hasil build_model.py:
  - dataset.pkl
  - embeddings.pkl
  - model_info.pkl
  - processed_dataset.xlsx

- templates/
  Berisi halaman HTML Flask:
  - index.html
  - hasil.html

- static/
  Berisi gambar, ikon, dan foto tempat.


3. CARA MENJALANKAN PROGRAM
---------------------------

Langkah 1 - Masuk ke folder project:

    cd "D:\SEFIN\PROJECT S (Lomba dll)\Project Machine Learning (2)\Project Machine Learning"

Langkah 2 - Install library:

    pip install -r requirements.txt

Langkah 3 - Build model dan dataset:

    python build_model.py

Output yang dihasilkan:

    models/dataset.pkl
    models/embeddings.pkl
    models/model_info.pkl
    models/processed_dataset.xlsx

Langkah 4 - Jalankan web:

    python app.py

Langkah 5 - Buka browser:

    http://127.0.0.1:5000


4. BACKEND REKOMENDASI: SBERT DAN TF-IDF
----------------------------------------

Default engine sekarang memakai SBERT/SentenceTransformer.

Artinya:

- build_model.py membuat embeddings.pkl dari model sentence-transformer.
- engine.py memakai embeddings.pkl tersebut untuk menghitung cosine similarity.
- Hasil pencarian lebih semantik, jadi query seperti "tempat sejarah",
  "nongkrong estetik", atau "kuliner legendaris" lebih mudah dicocokkan dengan
  deskripsi tempat.

Model default:

    paraphrase-multilingual-mpnet-base-v2

Jika ingin mencoba mode ringan TF-IDF karena laptop terasa berat, jalankan:

    $env:LOCAFIND_EMBEDDING_BACKEND="tfidf"
    python app.py

Jika ingin kembali ke mode semantik SBERT:

    $env:LOCAFIND_EMBEDDING_BACKEND="sbert"
    python app.py

Kalau environment variable tidak di-set, default-nya tetap:

    sbert


5. ALUR PROGRAM
---------------

Alur besar sistem:

1. User membuka halaman Home.

2. User mengisi form:
   - Hari Kunjungan
   - Budget Wisata
   - Jam Mulai
   - Jam Selesai
   - Kategori
   - Preferensi Tempat

3. Form dikirim ke route POST /hasil di app.py.

4. app.py mengambil input user, lalu memanggil:

      LocaFindEngine.plan_trip(...)

5. engine.py memproses input:
   - Memecah narasi preferensi menjadi beberapa aktivitas.
   - Menebak kategori dari kata kunci user.
   - Memfilter tempat berdasarkan kategori, hari buka, dan budget.
   - Menghitung similarity antara preferensi user dan data tempat.
   - Mengambil kandidat paling relevan.
   - Menyusun itinerary berdasarkan jam mulai, jam selesai, durasi kunjungan,
     estimasi waktu tempuh, dan sisa budget.

6. Hasil itinerary dikirim ke templates/hasil.html.

7. Browser menampilkan:
   - Ringkasan itinerary
   - Daftar tempat rekomendasi
   - Jadwal perjalanan
   - Peta rute
   - Evaluasi budget, waktu, jarak, dan kualitas rekomendasi


6. KAPAN HARUS MENJALANKAN ULANG build_model.py?
------------------------------------------------

Jalankan ulang build_model.py jika:

- Dataset LocaFind.xlsx berubah.
- Kolom dataset diperbaiki atau ditambah.
- MODEL_NAME di build_model.py diganti.
- File di folder models/ terhapus.

Setelah build ulang, jalankan kembali:

    python app.py


7. TROUBLESHOOTING
------------------

Masalah:
File models/embeddings.pkl tidak ditemukan.

Solusi:

    python build_model.py


Masalah:
Model SBERT gagal dimuat dari cache lokal.

Solusi:
Jalankan build_model.py dengan internet aktif terlebih dahulu supaya model
ter-download dan tersimpan di cache lokal.


Masalah:
App terasa berat atau lama saat request pertama.

Solusi:
Gunakan fallback TF-IDF:

    $env:LOCAFIND_EMBEDDING_BACKEND="tfidf"
    python app.py


Masalah:
Tidak ada itinerary yang cocok.

Solusi:
Coba longgarkan salah satu input:

- Tambah budget.
- Perpanjang durasi perjalanan.
- Pilih lebih banyak kategori.
- Gunakan preferensi yang lebih umum.
- Pastikan hari dan jam kunjungan cocok dengan jam buka tempat.


8. CATATAN PENTING
------------------

- Mode SBERT lebih akurat secara makna, tetapi lebih berat.
- Mode TF-IDF lebih ringan, tetapi lebih literal terhadap kata-kata.
- Untuk presentasi/demo kualitas rekomendasi, gunakan default SBERT.
- Untuk laptop spek rendah, gunakan TF-IDF sebagai mode cadangan.

