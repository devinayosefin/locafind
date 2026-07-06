import os
from flask import Flask, render_template, request, redirect, url_for, flash

from engine import LocaFindEngine, ALLOWED_CATEGORIES

app = Flask(__name__)
app.secret_key = "locafind-dev-secret"  # ganti kalau deploy ke production

# Default fallback: pusat Kota Surabaya, dipakai kalau deteksi IP gagal
# (umum terjadi saat testing di localhost / jaringan lokal).
FALLBACK_LAT, FALLBACK_LON = -7.2575, 112.7521

VALID_DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
CATEGORY_LABELS = {
    "wisata": "Wisata",
    "kuliner": "Kuliner",
    "cafe": "Cafe",
    "pusat perbelanjaan": "Pusat Perbelanjaan",
}
CATEGORY_OPTIONS = [{"value": c, "label": CATEGORY_LABELS[c]} for c in ALLOWED_CATEGORIES]

_engine = None


def get_engine():
    """Lazy-load engine (model embedding baru dimuat saat request pertama)."""
    global _engine
    if _engine is None:
        _engine = LocaFindEngine()
    return _engine


@app.route("/")
def index():
    return render_template("index.html", days=VALID_DAYS, category_options=CATEGORY_OPTIONS)


@app.route("/hasil", methods=["POST"])
def hasil():
    visit_day = request.form.get("visit_day", "Senin").strip()
    start_time = (request.form.get("start_time") or "08:00").strip()
    end_time = (request.form.get("end_time") or "21:00").strip()
    narration = request.form.get("narration", "").strip()

    selected_categories = [
        c for c in request.form.getlist("categories") if c in ALLOWED_CATEGORIES
    ]
    # Tidak ada yang dicentang, atau semua dicentang -> None (artinya cari di
    # semua kategori, biarkan tebakan kata kunci per-aktivitas yang menentukan).
    if 0 < len(selected_categories) < len(ALLOWED_CATEGORIES):
        forced_categories = selected_categories
    else:
        forced_categories = None

    try:
        budget = int(request.form.get("budget", 0) or 0)
    except ValueError:
        budget = 0

    if visit_day not in VALID_DAYS:
        visit_day = "Senin"

    engine = get_engine()

    lat, lon = engine.get_current_location()
    if lat is None:
        lat, lon = FALLBACK_LAT, FALLBACK_LON

    result = engine.plan_trip(
        narration=narration,
        visit_day=visit_day,
        start_time=start_time,
        end_time=end_time,
        budget=budget,
        lat=lat,
        lon=lon,
        forced_categories=forced_categories,
    )

    return render_template(
        "hasil.html",
        result=result,
        start_lat=lat,
        start_lon=lon,
    )


if __name__ == "__main__":
    app.run(debug=False)
