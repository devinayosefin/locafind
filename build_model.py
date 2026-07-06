import os
import re
import pickle
import pandas as pd

from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset default disimpan satu folder dengan script ini. Kalau dataset dipindah,
# bisa override lewat environment variable LOCAFIND_DATASET_PATH.
DATASET_PATH = os.environ.get(
    "LOCAFIND_DATASET_PATH",
    os.path.join(BASE_DIR, "Dataset LocaFind.xlsx"),
)

# Semua file hasil (dataset.pkl, embeddings.pkl, dst) disimpan di folder models.
# Bisa override lewat LOCAFIND_MODEL_DIR kalau diperlukan.
OUTPUT_DIR = os.environ.get(
    "LOCAFIND_MODEL_DIR",
    os.path.join(BASE_DIR, "models"),
)
SENTENCE_TRANSFORMER_DIR = os.path.join(OUTPUT_DIR, "sentence_transformer")

# Nama model sentence-transformer yang dipakai untuk embedding.
# Ganti baris ini saja kalau mau coba model lain, misalnya:
#   - "LazarusNLP/all-indobert-base-v4"                 -> khusus Bahasa Indonesia
#   - "paraphrase-multilingual-mpnet-base-v2"            -> multilingual, lebih besar & akurat
#   - "paraphrase-multilingual-MiniLM-L12-v2"            -> multilingual, lebih ringan/cepat
# Catatan: kalau model diganti, embeddings.pkl WAJIB di-generate ulang
# (jalankan ulang script ini) karena dimensi/representasi vektornya berbeda.
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

DAYS = [
    "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"
]

# Kata kunci tambahan per kategori, disuntikkan ke teks sebelum di-embed.
# Tujuannya supaya query kasual/gaul dari user (ngopi, nongkrong, healing, dst)
# tetap bisa match secara semantik walau tidak ada di Description asli.
# (Ide ini diadaptasi dari train_model.py milik teman, dirapikan jadi dictionary
# per kategori agar tidak hardcode satu string untuk semua kondisi.)
CATEGORY_SYNONYMS = {
    "kuliner": "makan enak kuliner legendaris hits rekomendasi",
    "cafe": "ngopi nongkrong santai cozy hangout estetik instagramable",
    "wisata": "healing piknik jalan-jalan liburan rekreasi seru",
    "pusat perbelanjaan": "belanja mall pusat oleh-oleh shopping",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# PRICE PARSER
# =====================================================

def parse_price(price):

    if pd.isna(price):
        return pd.Series([0, 0, 0])

    text = str(price).strip()

    if text == "":
        return pd.Series([0, 0, 0])

    text = text.replace(".", "")
    text = text.replace(",", "")

    nums = re.findall(r"\d+", text)
    nums = [int(x) for x in nums]

    if len(nums) == 0:
        return pd.Series([0, 0, 0])

    if len(nums) == 1:
        return pd.Series([nums[0], nums[0], nums[0]])

    low = min(nums)
    high = max(nums)
    avg = int((low + high) / 2)

    return pd.Series([low, high, avg])


# =====================================================
# COORDINATE PARSER
# =====================================================
# Dataset baru punya kolom "Coordinate" berformat bersih "lat, long"
# (mis. "-7.2598901, 112.7393419"), sedangkan kolom "Lat"/"Long" mentahnya
# sering rusak (koma dipakai sebagai pemisah ribuan oleh Excel/Sheets,
# misal "-7.2598901" berubah jadi "-72,598,901"). Supaya aman, koordinat
# akhir SELALU diturunkan ulang dari kolom "Coordinate", bukan dari kolom
# "Lat"/"Long" apa adanya.

def parse_coordinate(coord_text):

    if pd.isna(coord_text):
        return pd.Series([None, None])

    text = str(coord_text).strip()

    if text == "" or "," not in text:
        return pd.Series([None, None])

    parts = text.split(",")
    if len(parts) != 2:
        return pd.Series([None, None])

    try:
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
    except ValueError:
        return pd.Series([None, None])

    return pd.Series([lat, lon])


# =====================================================
# OPEN DAY EXPANDER
# =====================================================

def expand_open_days(day_text):

    if pd.isna(day_text):
        return DAYS

    text = str(day_text).strip()
    text = text.replace("–", "-")

    if "-" not in text:
        if text in DAYS:
            return [text]
        return DAYS

    parts = text.split("-")

    if len(parts) != 2:
        return DAYS

    start_day = parts[0].strip()
    end_day = parts[1].strip()

    if start_day not in DAYS or end_day not in DAYS:
        return DAYS

    start_idx = DAYS.index(start_day)
    end_idx = DAYS.index(end_day)

    if start_idx <= end_idx:
        return DAYS[start_idx:end_idx + 1]

    return DAYS[start_idx:] + DAYS[:end_idx + 1]


# =====================================================
# DURATION ESTIMATOR
# =====================================================

def estimate_duration(row):

    category = str(row["Category"]).lower()
    tags = str(row["Tags"]).lower()
    place = str(row["Place_Name"]).lower()

    if "museum" in tags:
        return 120
    if "monumen" in tags:
        return 90
    if "taman" in tags:
        return 90
    if "kebun binatang" in place:
        return 180

    if category == "wisata":
        return 120
    if category == "kuliner":
        return 60
    if category == "cafe":
        return 120
    if category == "pusat perbelanjaan":
        return 180

    return 90


# =====================================================
# TEXT FEATURE (dengan enrichment)
# =====================================================

def enrich_text(row):

    base = " ".join([
        str(row.get("Place_Name", "")),
        str(row.get("Category", "")),
        str(row.get("Tags", "")),
        str(row.get("Description", "")),
    ])

    category = str(row.get("Category", "")).lower()
    extra = CATEGORY_SYNONYMS.get(category, "")

    return f"{base} {extra}".strip()


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    print("\n====================================")
    print("LOCAFIND - BUILD MODEL")
    print("====================================")

    print("\nMemuat dataset...")
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.strip()
    print(f"Data awal : {len(df)}")

    # --- clean rating ---
    df["Rating"] = df["Rating"].astype(str).str.replace(",", ".", regex=False)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)

    # --- clean review ---
    df["Review_count"] = pd.to_numeric(df["Review_count"], errors="coerce").fillna(0)

    # --- koordinat: SELALU ambil dari kolom "Coordinate", timpa Lat/Long lama ---
    if "Coordinate" not in df.columns:
        raise ValueError(
            "Kolom 'Coordinate' tidak ditemukan di dataset. "
            "Kolom ini wajib ada (format 'lat, long') karena Lat/Long final "
            "diturunkan dari sini, bukan dari kolom Lat/Long mentah."
        )

    df[["Lat", "Long"]] = df["Coordinate"].apply(parse_coordinate)
    df = df.dropna(subset=["Lat", "Long"])
    print(f"Data setelah parsing koordinat dari kolom Coordinate : {len(df)}")

    # --- Google Maps URL: rapikan whitespace, kosongkan kalau memang kosong ---
    if "Google_Maps_URL" in df.columns:
        df["Google_Maps_URL"] = df["Google_Maps_URL"].astype(str).str.strip()
        df.loc[df["Google_Maps_URL"].isin(["", "nan", "None"]), "Google_Maps_URL"] = None
    else:
        df["Google_Maps_URL"] = None

    # --- price feature ---
    df[["min_price", "max_price", "price_value"]] = df["Price"].apply(parse_price)

    # --- expand visit day (1 baris per hari buka) ---
    expanded_rows = []
    for _, row in df.iterrows():
        for day in expand_open_days(row["Open_Day"]):
            temp = row.copy()
            temp["Visit_Day"] = day
            expanded_rows.append(temp)
    df = pd.DataFrame(expanded_rows)
    print(f"Data setelah expand hari : {len(df)}")

    # --- duration feature ---
    df["estimated_duration_min"] = df.apply(estimate_duration, axis=1)

    # --- text feature (dengan enrichment sinonim) ---
    df["combined_text"] = df.apply(enrich_text, axis=1)

    # --- final clean ---
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df["row_id"] = df.index  # untuk sinkronisasi index embedding
    print(f"Data final : {len(df)}")

    # --- load model & generate embeddings ---
    print(f"\nMemuat model embedding: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Membuat embeddings...")
    embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

    # --- save pkl ---
    with open(os.path.join(OUTPUT_DIR, "dataset.pkl"), "wb") as f:
        pickle.dump(df, f)

    with open(os.path.join(OUTPUT_DIR, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    # --- simpan info model yang dipakai, biar locafind.py bisa load model yang sama ---
    with open(os.path.join(OUTPUT_DIR, "model_info.pkl"), "wb") as f:
        pickle.dump(
            {
                "model_name": MODEL_NAME,
                "local_model_dir": "sentence_transformer",
            },
            f,
        )

    # --- simpan model ke folder project ---
    # Penting untuk deploy Render: saat runtime, engine memuat model dari folder
    # models/sentence_transformer agar tidak bergantung pada cache Hugging Face.
    model.save(SENTENCE_TRANSFORMER_DIR)

    # --- export excel (buat inspeksi manual) ---
    excel_columns = [
        "row_id",
        "Place_ID", "Place_Name", "Category", "Tags",
        "Price", "min_price", "max_price", "price_value",
        "Rating", "Review_count",
        "Open_Day", "Visit_Day", "Open_time", "Close_time",
        "Coordinate", "Lat", "Long",
        "Google_Maps_URL",
        "estimated_duration_min",
        "combined_text",
    ]
    available_columns = [c for c in excel_columns if c in df.columns]
    df[available_columns].to_excel(
        os.path.join(OUTPUT_DIR, "processed_dataset.xlsx"), index=False
    )

    print("\n====================================")
    print("BUILD MODEL SELESAI")
    print("====================================")
    print(f"Model embedding : {MODEL_NAME}")
    print(f"Total data      : {len(df)}")
    print(f"Tersimpan di folder '{OUTPUT_DIR}/':")
    print("  - dataset.pkl")
    print("  - embeddings.pkl")
    print("  - model_info.pkl")
    print("  - sentence_transformer/")
    print("  - processed_dataset.xlsx")


if __name__ == "__main__":
    main()
