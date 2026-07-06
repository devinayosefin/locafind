"""
locafind_app.py — CLI untuk ngetes engine LocaFind langsung dari terminal,
TANPA perlu jalanin Flask/browser. Berguna buat ngecek angka-angka mentah
(similarity/skor popularitas, rating, review, jarak, dst) sebelum dites lewat
web, dan buat verifikasi kalau perubahan di engine.py sudah sesuai harapan.

Script ini makai class LocaFindEngine yang SAMA PERSIS dengan yang dipakai
app.py — jadi hasil yang keluar di sini dijamin konsisten dengan hasil yang
bakal muncul di halaman /hasil.

Syarat sebelum jalan:
1. Sudah generate models/dataset.pkl, models/embeddings.pkl, models/model_info.pkl
   lewat `python build_model.py`.
2. Jalankan dari dalam folder locafind_web/ (biar import `engine` ketemu).
"""

from engine import LocaFindEngine, ALLOWED_CATEGORIES

CATEGORY_LABELS = {
    "wisata": "Wisata",
    "kuliner": "Kuliner",
    "cafe": "Cafe",
    "pusat perbelanjaan": "Pusat Perbelanjaan",
}


def rupiah(value):
    return f"Rp{value:,.0f}".replace(",", ".")


def ask_categories():
    print("\nPilih kategori (boleh lebih dari satu, pisahkan dengan koma).")
    print("Kosongkan (Enter) kalau mau cari di SEMUA kategori (sama kayak")
    print("checkbox Kategori di web dikosongkan semua).")
    for i, cat in enumerate(ALLOWED_CATEGORIES, start=1):
        print(f"  {i}. {CATEGORY_LABELS[cat]}")
    raw = input("Pilihan (mis. 1,3): ").strip()
    if not raw:
        return None
    picks = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(ALLOWED_CATEGORIES):
                picks.append(ALLOWED_CATEGORIES[idx])
    return picks or None


def print_itinerary(result):
    print("\n" + "=" * 80)
    print("ITINERARY")
    print("=" * 80)

    if not result["places"]:
        print("Tidak ada itinerary yang cocok dengan kombinasi hari/jam/budget/kategori ini.")
        print("Coba longgarkan budget, ganti jam, atau kosongkan filter kategori.")
        return

    for i, p in enumerate(result["places"], start=1):
        print(f"\n[{i}] {p['name']}  ({p['category']})")
        print(f"    Jam kunjungan    : {p['arrive']} - {p['leave']}  (buka {p['open_time']}-{p['close_time']})")
        print(f"    Durasi kunjungan : {p['duration_min']} menit")
        print(f"    Biaya            : {rupiah(p['cost'])}")
        print(f"    Rating           : {p['rating']}  (jumlah review: {p['review_count']})")
        print(f"    Similarity/skor  : {p['similarity']:.3f}")
        print(f"    Jarak dr sblmnya : {p['distance_km']} km")

    print("\n--- Rekomendasi teratas (top 5 by similarity/skor) ---")
    for i, p in enumerate(result["top_recommendations"], start=1):
        print(f"  {i}. {p['name']}  (skor: {p['similarity']:.3f})")


def print_summary(result):
    print("\n" + "=" * 80)
    print("TRIP SUMMARY")
    print("=" * 80)
    mode_label = "AUTO — sistem yang menentukan tempat & urutannya" \
        if result["order_mode"] == "auto" else "MANUAL — mengikuti urutan preferensi kamu"
    print(f"Mode urutan         : {mode_label}")
    print(f"Jumlah tempat       : {result['place_count']}")
    print(f"Budget terpakai     : {rupiah(result['used_budget'])} / {rupiah(result['budget'])}")
    print(f"Sisa budget         : {rupiah(result['remaining_budget'])}")
    print(f"Waktu selesai       : {result['finish_time']}")
    print(f"Total jarak tempuh  : {result['total_distance']} km")
    print(f"Rata-rata similarity: {result['avg_similarity']}")
    print(f"Kualitas rekomendasi: {result['quality']}")
    print(f"Status budget       : {result['budget_status']}")
    print(f"Status jam          : {result['time_status']}")
    if result["order_mode"] == "auto":
        print("\nCatatan: di mode AUTO, kolom 'Similarity/skor' di atas BUKAN")
        print("kecocokan semantik ke preferensi (karena memang tidak ada preferensi")
        print("teks) — itu skor popularitas = 0.6*rating_norm + 0.4*review_norm.")


def main():
    print("\n====================================")
    print("        LOCAFIND - CLI DEBUG")
    print("====================================")
    print("(Pakai engine.py yang sama dengan app.py Flask)")

    engine = LocaFindEngine()

    print("\nMendeteksi lokasi user lewat IP...")
    lat, lon = engine.get_current_location()
    if lat is not None:
        print(f"Lokasi terdeteksi: {lat:.6f}, {lon:.6f}")
    else:
        print("Deteksi lokasi gagal (umum kalau di jaringan lokal/kampus).")
        lat = float(input("Latitude  : "))
        lon = float(input("Longitude : "))

    print("\n========== INPUT ==========")
    visit_day = input("Hari kunjungan (Senin/Selasa/.../Minggu) [Senin]: ").strip() or "Senin"
    start_time = input("Jam mulai (HH:MM) [08:00]: ").strip() or "08:00"
    end_time = input("Jam selesai (HH:MM) [21:00]: ").strip() or "21:00"
    budget_raw = input("Budget total (Rp) [160000]: ").strip() or "160000"

    try:
        budget = int(budget_raw)
    except ValueError:
        print("Budget tidak valid, pakai default 160000.")
        budget = 160000

    categories = ask_categories()

    print("\nCeritakan preferensi tempatmu, ATAU tulis 'terserah'/'bebas'/'rekomendasikan")
    print("saja' kalau mau urutan & pilihan tempat diserahkan penuh ke sistem:")
    narration = input("Preferensi: ").strip()

    print("\nMenyusun itinerary...")
    result = engine.plan_trip(
        narration=narration,
        visit_day=visit_day,
        start_time=start_time,
        end_time=end_time,
        budget=budget,
        lat=lat,
        lon=lon,
        forced_categories=categories,
    )

    print_itinerary(result)
    print_summary(result)


if __name__ == "__main__":
    main()