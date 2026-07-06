# LocaFind Web

Frontend (`index.html`, `hasil.html`) sudah disambungkan ke engine rekomendasi
(`build_model.py` + `engine.py`, hasil refactor dari `locafind_app.py`) lewat
Flask (`app.py`). Login/register dihapus sesuai permintaan — situs hanya
punya 2 halaman: **Home** (form pencarian) dan **Hasil** (itinerary).

## Struktur folder

```
locafind_web/
  app.py                  <- Flask server (routing + koneksi ke engine)
  engine.py               <- LocaFindEngine (refactor locafind_app.py, tanpa CLI/folium)
  build_model.py          <- Script offline untuk generate models/*.pkl
  requirements.txt
  models/                 <- HARUS diisi dengan hasil build_model.py
  templates/
    index.html            <- Halaman form pencarian
    hasil.html             <- Halaman hasil itinerary (di-render dinamis via Jinja)
  static/
    images/                <- hero.jpg, footer.jpg, wisata.jpg, kuliner.jpg, cafe.jpg, mall.jpg, dst.
```

## Cara menjalankan

1. **Install dependency**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate model & dataset** (wajib dijalankan dulu, sekali saja / setiap dataset berubah)

   Buka `build_model.py`, set `DATASET_PATH` ke lokasi `Dataset_LocaFind.xlsx` kamu,
   lalu pastikan `OUTPUT_DIR = "models"` menunjuk ke folder `locafind_web/models`
   (atau jalankan script ini dari dalam folder `locafind_web/`).
   ```bash
   python build_model.py
   ```
   Ini akan membuat `models/dataset.pkl`, `models/embeddings.pkl`, `models/model_info.pkl`.

3. **Jalankan web app**
   ```bash
   python app.py
   ```
   Buka `http://127.0.0.1:5000` di browser.

## Alur data (front end -> engine)

1. User isi form di `index.html`: **Hari Kunjungan**, **Kategori**, **Budget**,
   **Jam mulai/selesai**, **Preferensi Tempat** (narasi bebas).
   - Field **Hari Kunjungan** baru ditambahkan di form karena `engine.py`
     butuh `visit_day` untuk memfilter tempat yang buka hari itu
     (sebelumnya tidak ada di desain aslinya).
2. Form di-submit lewat `POST /hasil` ke `app.py`.
3. `app.py` memanggil `LocaFindEngine.plan_trip(...)`, yang di dalamnya:
   - `extract_intents()` memecah narasi jadi daftar aktivitas,
   - `guess_category()` / kategori paksa dari dropdown menentukan kategori tiap aktivitas,
   - `build_itinerary()` menyusun jadwal (cek jam buka/tutup, waktu tempuh, budget),
   - `evaluate()` menghitung metrik (similarity, distance, time utilization).
4. Hasilnya dirender langsung ke `hasil.html` (tabel timeline, kartu tempat,
   peta Leaflet, ringkasan trip) — semua data sudah dinamis, tidak ada lagi
   angka hardcode.

## Ikon yang perlu kamu tambahkan

Semua emoji sudah dihapus dari `index.html` dan `hasil.html`, diganti `<img>`
yang path-nya menunjuk ke `static/images/icons/`. Tinggal taruh file dengan
nama berikut di folder itu (PNG/SVG apa saja, ukuran render sudah diatur
lewat class Tailwind `w-4 h-4` / `w-6 h-6` / `w-8 h-8` jadi tidak perlu pas-pasan):

| File | Dipakai di |
|---|---|
| `icon-calendar.png` | Field "Hari Kunjungan" |
| `icon-budget.png` | Field "Budget Wisata" |
| `icon-clock.png` | Field "Durasi Perjalanan" (start & end) |
| `icon-smart.png` | Kartu fitur "Smart Recommendations" |
| `icon-itinerary.png` | Kartu fitur "Automatic Itinerary" |
| `icon-route.png` | Kartu fitur "Optimized Route" |
| `icon-wisata.png` | Placeholder foto kartu hasil, kategori Wisata |
| `icon-kuliner.png` | Placeholder foto kartu hasil, kategori Kuliner |
| `icon-cafe.png` | Placeholder foto kartu hasil, kategori Cafe |
| `icon-mall.png` | Placeholder foto kartu hasil, kategori Pusat Perbelanjaan |

Sebelum file-nya ada, browser cuma menampilkan ikon gambar rusak (kotak kecil)
di tempat itu — tidak akan error, tampilan lain tetap jalan normal.

## Kategori dihapus dari form — sekarang AI yang menebak sendiri

Dropdown "Kategori" sudah dihapus dari `index.html`. Alasannya: memaksa satu
kategori ke semua aktivitas dalam narasi menurunkan kualitas hasil (mis. pilih
"Wisata" padahal narasi juga minta "makan bebek goreng" → sistem tetap
maksa cari kuliner di kategori wisata, jadi nggak nyambung).

Sekarang alurnya di `engine.py`:
1. Narasi dipecah per klausa (koma/titik) lewat `extract_intents()`.
2. Klausa yang **tidak mengandung kata kunci kategori apa pun**
   (wisata/kuliner/cafe/mall) dibuang — ini yang dulu menyebabkan hasil asal
   pilih dengan similarity rendah, karena kalimat filler seperti "aku pengen
   liburan dari pagi sampe sore" dulu tetap dipaksa jadi 1 slot itinerary.
3. Kalau setelah difilter tidak ada klausa tersisa sama sekali (narasi memang
   generik total, misalnya "buatkan itinerary 1 hari budget 100000"), sistem
   otomatis pakai kombinasi seimbang: 1 wisata + 1 kuliner + 1 cafe.
4. Batasan harga per-aktivitas juga sekarang dibaca dari kalimatnya sendiri,
   misalnya *"cafe yang budget nya dibawah 50000"* akan membatasi pencarian
   cafe itu maksimal Rp50.000 (terlepas dari sisa budget total masih lebih
   besar dari itu).

Kalau nanti masih ada kombinasi kalimat yang salah tebak kategorinya, kata
kunci per kategori ada di `CATEGORY_KEYWORDS` (dekat atas `engine.py`) —
tinggal tambahkan kata baru di situ.

## Catatan tentang foto tempat

Dataset belum punya URL/foto per tempat, jadi kartu di halaman hasil
menampilkan placeholder ikon + teks **"Foto menyusul"** untuk setiap tempat
(bukan foto asal-cocok). Kalau nanti dataset punya kolom foto (misalnya
`Image_URL`), tinggal:
- tambahkan kolom itu ke `excel_columns` di `build_model.py`,
- di `engine.py` method `_serialize_itinerary`, ganti `"icon": self._pick_icon(...)`
  jadi `"image": row["Image_URL"]`,
- di `hasil.html`, ganti div placeholder jadi `<img src="{{ p.image }}">`.

Foto di halaman **awal** (hero, footer, 4 kategori) tetap memakai file yang
sudah kamu upload di `static/images/` (`hero.jpg`, `footer.jpg`, `wisata.jpg`,
`kuliner.jpg`, `cafe.jpg`, `mall.jpg`). Kalau mau ganti dengan foto lain
(misalnya foto Patung Suroboyo untuk hero), tinggal timpa file tersebut
dengan nama yang sama.

## Kalau tidak ada hasil

Jika kombinasi hari/jam/budget/preferensi tidak menghasilkan itinerary sama
sekali, `hasil.html` menampilkan pesan "Belum ada itinerary yang cocok"
dengan tombol kembali ke halaman pencarian (lebih baik daripada halaman kosong).
