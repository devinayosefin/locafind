import os
import re
import math
import pickle
from datetime import datetime, timedelta

import pandas as pd
import requests
# CATATAN: sentence_transformers SENGAJA tidak di-import di sini (top-level).
# Import-nya ditunda ke dalam _load_model_offline_first() supaya HF_HUB_OFFLINE
# bisa di-set SEBELUM huggingface_hub pertama kali di-import -- library itu
# cuma membaca env var ini SEKALI, waktu pertama diimpor, lalu "membekukannya"
# jadi konstanta. Kalau kita set env var-nya setelah modul ini sudah diimpor
# duluan (mis. lewat import di baris paling atas file), settingnya jadi
# terlambat dan tidak berpengaruh -- itu sebabnya warning "unauthenticated
# requests to the HF Hub" masih muncul walau HF_HUB_OFFLINE sudah di-set.
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RUNTIME_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".runtime_cache"
)
EMBEDDING_BACKEND_ENV = "LOCAFIND_EMBEDDING_BACKEND"
DEFAULT_EMBEDDING_BACKEND = "sbert"
SUPPORTED_EMBEDDING_BACKENDS = {"sbert", "tfidf"}
LOCAL_SENTENCE_TRANSFORMER_DIRNAME = "sentence_transformer"

# Folder foto asli per tempat, nama file mengikuti Place_ID (mis. "1.webp",
# "2.jpg") — ekstensi antar tempat boleh berbeda-beda karena foto diambil dari
# sumber yang bervariasi. Ditaruh di static/static_1/ (bukan static/images/)
# supaya tidak bentrok dengan aset UI (hero, ikon, dsb).
STATIC_PHOTO_DIRNAME = "static_1"
STATIC_PHOTO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "static", STATIC_PHOTO_DIRNAME
)
PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

# Kalau seluruh narasi generik (tidak ada kata kunci kategori sama sekali),
# pakai kombinasi seimbang ini supaya tetap dapat itinerary lintas-kategori.
DEFAULT_FALLBACK_ACTIVITIES = ["wisata rekreasi menarik", "makan kuliner khas enak", "cafe santai nongkrong"]

# 4 kategori yang valid di dataset (dipakai untuk filter checkbox & validasi).
ALLOWED_CATEGORIES = ["wisata", "kuliner", "cafe", "pusat perbelanjaan"]

# Kalau user menulis kalimat seperti "terserah" / "bebas" / "rekomendasikan saja"
# di kolom preferensi, itu artinya urutan itinerary diserahkan penuh ke sistem
# (bukan mengikuti urutan penyebutan aktivitas di kalimat).
AUTO_ORDER_TRIGGERS = [
    "terserah", "bebas", "rekomendasikan", "rekomendasi saja", "sesuai rekomendasi",
    "apa saja", "apa aja", "sesuai sistem", "biar sistem", "sistem yang atur",
    "sistem aja", "up to you", "random", "asal seru", "asal enak", "asal ramai",
    "sesukanya sistem", "diserahkan ke sistem",
]

# Query generik per kategori untuk mode "auto order" — sengaja meniru kata kunci
# yang sama dengan CATEGORY_SYNONYMS di build_model.py supaya similarity-nya
# tinggi terhadap tempat-tempat representatif kategori tsb, lalu rating/review
# yang menentukan urutan akhirnya.
AUTO_QUERY_BY_CATEGORY = {
    "kuliner": "makan enak kuliner legendaris hits rekomendasi",
    "cafe": "ngopi nongkrong santai cozy hangout estetik instagramable",
    "wisata": "healing piknik jalan-jalan liburan rekreasi seru",
    "pusat perbelanjaan": "belanja mall pusat oleh-oleh shopping",
}

# Menangkap batasan harga per-aktivitas dari kalimat bebas, misalnya:
# "cafe yang budget nya dibawah 50000", "maksimal 30rb", "budget 20.000"
PRICE_CONSTRAINT_RE = re.compile(
    r"(?:dibawah|di bawah|maksimal|maks|max|budget(?:nya)?|kurang dari)\s*(?:rp\.?\s*)?"
    r"([\d.,]+)\s*(rb|ribu|k|juta)?",
    re.IGNORECASE,
)

# Kata kunci untuk menebak kategori dari kalimat bebas user.
CATEGORY_KEYWORDS = {
    "cafe": ["cafe", "kafe", "ngopi", "kopi", "nongkrong"],
    "kuliner": ["makan", "kuliner", "resto", "restoran", "warung"],
    "wisata": ["wisata", "healing", "piknik", "jalan-jalan", "liburan", "sejarah", "museum", "taman"],
    "pusat perbelanjaan": ["belanja", "mall", "shopping"],
}

# Ikon fallback per kategori untuk placeholder "Foto menyusul" di kartu hasil
# (dataset belum punya foto asli per tempat). Filenya tinggal ditaruh di
# static/images/icons/ dengan nama yang sama.
CATEGORY_ICON = {
    "wisata": "icon-wisata.png",
    "kuliner": "icon-kuliner.png",
    "cafe": "icon-cafe.png",
    "pusat perbelanjaan": "icon-mall.png",
}

# =====================================================
# PARAMETER ALUR ALGORITMA (lihat diagram flowchart sistem)
# =====================================================

# Similarity adalah satu-satunya penentu kandidat yang lolos tahap relevansi.
# Rating, jumlah review, dan jarak baru dipakai untuk memilih di antara kandidat
# Top-N yang sudah relevan tersebut.
TOP_N_PER_CATEGORY = 10

# Bobot pemilihan sekunder di dalam shortlist relevan. Nilai ini sama sekali
# tidak memengaruhi siapa yang masuk Top-10 per kategori.
SECONDARY_WEIGHT_RATING = 0.30
SECONDARY_WEIGHT_REVIEW = 0.30
SECONDARY_WEIGHT_DISTANCE = 0.40

# Greedy Nearest-Neighbor Selection: maksimal 10x percobaan (dari kandidat
# terdekat ke yang lebih jauh) sebelum sebuah aktivitas ditandai "gagal
# mencari" dan dilewati.
MAX_GREEDY_RETRY = 10

# Radius bumi (km) untuk rumus Haversine.
EARTH_RADIUS_KM = 6371.0

# Set True supaya setiap kali sistem mencari kandidat, rincian skornya
# (similarity, rating_norm, review_norm, distance_norm, dan skor sekunder)
# ditampilkan di terminal (VSCode) tempat `python app.py`
# dijalankan. Set False kalau mau mematikan log ini lagi (misalnya saat
# sudah deploy ke production).
DEBUG_SCORES = True


class LocaFindEngine:

    def __init__(self, model_dir=MODEL_DIR):
        with open(os.path.join(model_dir, "dataset.pkl"), "rb") as f:
            self.df = pickle.load(f)

        model_info = self._read_model_info(model_dir)
        model_name = model_info.get("model_name", "paraphrase-multilingual-mpnet-base-v2")
        self.embedding_backend = self._selected_embedding_backend()

        if self.embedding_backend == "sbert":
            self._load_sbert_backend(model_dir, model_name, model_info)
        else:
            self._load_tfidf_backend()

    @staticmethod
    def _read_model_info(model_dir):
        model_info_path = os.path.join(model_dir, "model_info.pkl")
        if not os.path.exists(model_info_path):
            return {"model_name": "paraphrase-multilingual-mpnet-base-v2"}

        with open(model_info_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _selected_embedding_backend():
        """Pilih backend similarity.

        Default sekarang SBERT supaya runtime memakai embedding semantik yang
        dibuat oleh build_model.py. TF-IDF tetap bisa dipakai sebagai mode
        ringan dengan:
            $env:LOCAFIND_EMBEDDING_BACKEND="tfidf"
        """
        backend = os.environ.get(
            EMBEDDING_BACKEND_ENV, DEFAULT_EMBEDDING_BACKEND
        ).strip().lower()

        if backend not in SUPPORTED_EMBEDDING_BACKENDS:
            print(
                f"Backend embedding '{backend}' tidak dikenali. "
                f"Menggunakan default: {DEFAULT_EMBEDDING_BACKEND}."
            )
            return DEFAULT_EMBEDDING_BACKEND

        return backend

    def _load_sbert_backend(self, model_dir, model_name, model_info):
        embeddings_path = os.path.join(model_dir, "embeddings.pkl")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                "File models/embeddings.pkl tidak ditemukan. "
                "Jalankan `python build_model.py` terlebih dahulu."
            )

        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

        local_model_dir = model_info.get(
            "local_model_dir", LOCAL_SENTENCE_TRANSFORMER_DIRNAME
        )
        local_model_path = os.path.join(model_dir, local_model_dir)
        model_source = local_model_path if os.path.isdir(local_model_path) else model_name

        print(f"Memuat model embedding SBERT: {model_source} ...")
        self.model = self._load_model_offline_first(model_source)
        self.embedding_backend = "sbert"
        print("Model similarity SBERT siap.")

    def _load_tfidf_backend(self):
        print("Menyiapkan model similarity TF-IDF ...")
        self.embedding_backend = "tfidf"
        self.model = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=30000,
        )
        corpus = self.df["combined_text"].fillna("").astype(str)
        self.text_embeddings = self.model.fit_transform(corpus)
        print("Model similarity TF-IDF siap.")

    @staticmethod
    def _load_model_offline_first(model_name):
        """Load SentenceTransformer dari cache lokal Hugging Face saja
        (HF_HUB_OFFLINE=1), supaya TIDAK ada request ke internet sama sekali
        -- ini yang menghilangkan warning "unauthenticated requests to the
        HF Hub" dan mempercepat startup (nggak nunggu network round-trip
        tiap kali Flask debug reloader restart proses).

        PENTING: env var HF_HUB_OFFLINE di-set DI SINI, SEBELUM baris
        `from sentence_transformers import SentenceTransformer` dijalankan.
        `huggingface_hub` cuma membaca env var ini SEKALI, waktu pertama kali
        modulnya diimpor di seluruh proses Python, lalu membekukannya jadi
        konstanta -- makanya import sentence_transformers TIDAK ada di baris
        paling atas file ini (kalau ada di sana, dia akan diimpor duluan
        sebelum kode ini jalan, dan setting env var jadi terlambat/tidak
        berpengaruh, seperti yang terjadi di percobaan sebelumnya).
        """
        # Beberapa instalasi Windows menyisakan TEMP/TMP yang menunjuk ke folder
        # tidak valid. Import PyTorch membutuhkan direktori temporary yang bisa
        # ditulis; jika tidak, request /hasil gagal saat engine pertama dimuat.
        os.makedirs(RUNTIME_CACHE_DIR, exist_ok=True)
        os.environ["TEMP"] = RUNTIME_CACHE_DIR
        os.environ["TMP"] = RUNTIME_CACHE_DIR
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
            RUNTIME_CACHE_DIR, "torchinductor"
        )
        os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            # Import PERTAMA KALI terjadi di sini, setelah env var di-set.
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Gagal memuat model '{model_name}' dari cache lokal (mode offline).\n"
                f"Detail error asli: {exc}"
            ) from exc

    # ---------------------------------------------------
    # LOKASI & WAKTU TEMPUH
    # ---------------------------------------------------

    @staticmethod
    def get_current_location():
        try:
            response = requests.get("http://ip-api.com/json/", timeout=5)
            data = response.json()
            if data.get("status") == "success":
                return float(data["lat"]), float(data["lon"])
        except Exception:
            pass
        return None, None

    @staticmethod
    def estimate_travel_time(distance_km):
        minutes_per_km = 5  # asumsi kecepatan motor
        return int(distance_km * minutes_per_km)

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Perhitungan Jarak (Haversine Formula) — jarak garis lurus antar
        2 titik koordinat di permukaan bumi, dalam kilometer.
        """
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

    @staticmethod
    def _min_max_scale(series):
        """Normalisasi Min-Max Scaling: (x - min) / (max - min).

        Kalau semua nilai sama (max == min), kembalikan 1.0 untuk semuanya
        supaya tidak divide-by-zero dan tidak menjatuhkan skor siapa pun.
        """
        min_v = series.min()
        max_v = series.max()
        if max_v == min_v:
            return pd.Series([1.0] * len(series), index=series.index)
        return (series - min_v) / (max_v - min_v)

    @staticmethod
    def _print_score_table(query, guessed_category, categories, result_df):
        """Cetak ranking relevansi dan skor sekunder ke terminal.

        Ditampilkan per pemanggilan search_candidates (artinya per aktivitas/
        klausa narasi), supaya kelihatan kandidat mana yang menang dan
        similarity menentukan shortlist, sedangkan rating/review/distance hanya
        menjadi pembanding di dalam shortlist tersebut.
        """
        print("\n" + "=" * 100)
        print(f"[DEBUG SKOR] query='{query}'  kategori_tebakan={guessed_category}  filter_kategori={categories}")
        print("=" * 100)

        if result_df.empty:
            print("  (tidak ada kandidat yang lolos hard-constraint filtering)")
            return

        header = (
            f"{'Place_Name':<45}{'Category':<20}{'Sim':>7}{'Rate_N':>8}"
            f"{'Rev_N':>8}{'Dist_N':>8}{'SECOND':>9}"
        )
        print(header)
        print("-" * len(header))
        for _, r in result_df.iterrows():
            print(
                f"{str(r['Place_Name'])[:44]:<45}{str(r['Category'])[:19]:<20}"
                f"{r['similarity']:>7.3f}{r['rating_norm']:>8.3f}"
                f"{r['review_norm']:>8.3f}{r['distance_norm']:>8.3f}"
                f"{r['secondary_score']:>9.3f}"
            )
        print()

    # ---------------------------------------------------
    # EKSTRAKSI AKTIVITAS DARI NARASI BEBAS
    # ---------------------------------------------------

    @staticmethod
    def extract_intents(narration):
        segments = re.split(r"[,.]", narration or "")
        return [seg.strip() for seg in segments if seg.strip()]

    @staticmethod
    def guess_category(activity_text):
        text = activity_text.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(k in text for k in keywords):
                return category
        return None

    @staticmethod
    def _extract_price_cap(activity_text):
        """Tangkap batasan harga per-aktivitas, misal 'cafe dibawah 50000'."""
        match = PRICE_CONSTRAINT_RE.search(activity_text.lower())
        if not match:
            return None
        number_str = re.sub(r"[.,]", "", match.group(1))
        if not number_str.isdigit():
            return None
        value = int(number_str)
        unit = match.group(2)
        if unit in ("rb", "ribu", "k"):
            value *= 1000
        elif unit == "juta":
            value *= 1_000_000
        return value

    # ---------------------------------------------------
    # PENCARIAN KANDIDAT (similarity-first)
    # ---------------------------------------------------

    def search_candidates(self, query, visit_day, remaining_budget, current_lat, current_lon,
                           categories=None, guessed_category=None,
                           top_n_per_category=TOP_N_PER_CATEGORY):
        """Pipeline pencarian kandidat dengan similarity sebagai gerbang relevansi:

        1. Hard Constraint Filtering (Price <= Remaining Budget)
        2. Semantic Search (SBERT + Cosine Similarity)          -> Skor_AI
        3. Ranking murni berdasarkan similarity
        4. Ambil Top-10 per kategori
        5. Di dalam shortlist relevan, hitung skor sekunder dari rating,
           jumlah review, dan jarak

        Skor sekunder tidak pernah memengaruhi kandidat yang masuk Top-10.
        """

        candidate = self.df[self.df["Visit_Day"].str.lower() == visit_day.lower()].copy()

        if categories:
            allowed = [c.lower() for c in categories]
            candidate = candidate[candidate["Category"].str.lower().isin(allowed)]

        # --- 1. Hard Constraint Filtering ---
        candidate = candidate[candidate["price_value"] <= remaining_budget]
        candidate = candidate.dropna(subset=["Lat", "Long"])

        if candidate.empty:
            return candidate

        # --- 2. Semantic Search (Skor_AI) ---
        if getattr(self, "embedding_backend", "sbert") == "tfidf":
            query_emb = self.model.transform([query])
            cand_emb = self.text_embeddings[candidate["row_id"].values]
            candidate["similarity"] = cosine_similarity(query_emb, cand_emb)[0]
        else:
            query_emb = self.model.encode(query)
            cand_emb = self.embeddings[candidate["row_id"].values]
            candidate["similarity"] = cosine_similarity([query_emb], cand_emb)[0]

        # --- 3. Perhitungan Jarak (belum memengaruhi ranking relevansi) ---
        candidate["distance"] = candidate.apply(
            lambda r: self.haversine_distance(
                current_lat, current_lon, float(r["Lat"]), float(r["Long"])
            ),
            axis=1,
        )

        # --- 4. Gerbang relevansi: ranking MURNI similarity, Top-10/kategori ---
        candidate = candidate.sort_values("similarity", ascending=False)
        candidate = candidate.groupby(
            candidate["Category"].str.lower(), group_keys=False
        ).head(top_n_per_category)

        # --- 5. Skor sekunder, hanya di antara kandidat yang sudah relevan ---
        candidate["rating_norm"] = self._min_max_scale(candidate["Rating"])
        candidate["review_norm"] = self._min_max_scale(candidate["Review_count"])
        candidate["distance_norm"] = 1 - self._min_max_scale(candidate["distance"])
        candidate["secondary_score"] = (
            SECONDARY_WEIGHT_RATING * candidate["rating_norm"]
            + SECONDARY_WEIGHT_REVIEW * candidate["review_norm"]
            + SECONDARY_WEIGHT_DISTANCE * candidate["distance_norm"]
        )

        # Kompatibilitas untuk konsumen lama: score kini identik dengan
        # similarity dan bukan lagi blended score.
        candidate["score"] = candidate["similarity"]
        result = candidate.sort_values("similarity", ascending=False)

        if DEBUG_SCORES:
            self._print_score_table(query, guessed_category, categories, result)

        return result

    def search_candidates_popularity(self, visit_day, remaining_budget, categories=None, top_n=30):
        """Sama seperti search_candidates, tapi TANPA embedding teks sama sekali.

        Dipakai kalau user tidak memberi preferensi spesifik ('terserah', 'bebas',
        dst) — karena tidak ada preferensi nyata untuk dicocokkan, similarity
        semantik tidak relevan dan cuma menambah noise. Ranking di sini murni
        dari rating & jumlah review.
        """

        candidate = self.df[self.df["Visit_Day"].str.lower() == visit_day.lower()].copy()

        if categories:
            allowed = [c.lower() for c in categories]
            candidate = candidate[candidate["Category"].str.lower().isin(allowed)]

        candidate = candidate[candidate["price_value"] <= remaining_budget]
        candidate = candidate.dropna(subset=["Lat", "Long"])

        if candidate.empty:
            return candidate

        max_rating = candidate["Rating"].max() or 1
        max_review = candidate["Review_count"].max() or 1
        candidate["rating_norm"] = candidate["Rating"] / max_rating
        candidate["review_norm"] = candidate["Review_count"] / max_review

        candidate["score"] = 0.6 * candidate["rating_norm"] + 0.4 * candidate["review_norm"]
        # Field "similarity" diisi skor popularitas ini (bukan kecocokan semantik)
        # supaya kode serialize/evaluate yang sudah ada tetap jalan tanpa
        # perubahan tambahan.
        candidate["similarity"] = candidate["score"]

        return candidate.sort_values("score", ascending=False).head(top_n)

    # ---------------------------------------------------
    # PENYUSUNAN ITINERARY (time + budget + duration aware)
    # ---------------------------------------------------

    def build_itinerary(self, activities, visit_day, start_time, end_time,
                         budget, start_lat, start_lon, forced_categories=None):

        current_time = datetime.strptime(start_time, "%H:%M")
        limit_time = datetime.strptime(end_time, "%H:%M")

        current_location = (start_lat, start_lon)
        used_budget = 0
        itinerary = []
        visited_names = set()
        all_candidate_similarities = []

        for activity in activities:

            # --- Decision: "Waktu, Anggaran, dan Kategori Masih Tersedia?" ---
            # Kalau jam sudah lewat batas, atau budget sudah habis, berhenti di
            # sini juga (lanjut ke Visualisasi Itinerary) daripada tetap mencoba
            # aktivitas berikutnya yang sudah pasti tidak akan muat.
            remaining_budget = budget - used_budget
            if current_time >= limit_time or remaining_budget <= 0:
                break

            price_cap = self._extract_price_cap(activity)
            effective_budget = remaining_budget if price_cap is None else min(remaining_budget, price_cap)

            guessed = self.guess_category(activity)

            if forced_categories:
                # User sudah membatasi kategori lewat filter checkbox. Kalau kalimat
                # aktivitas ini kebetulan sebut kategori yang juga ada di daftar itu,
                # persempit lagi ke kategori spesifik itu saja; kalau tidak nyambung
                # sama sekali, tetap cari di dalam kategori-kategori yang dipilih user.
                categories = [guessed] if guessed in forced_categories else forced_categories
            else:
                categories = [guessed] if guessed else None

            # --- Filter kategori/budget -> ranking similarity murni -> Top-10
            #     per kategori -> penilaian sekunder di dalam shortlist ---
            candidates = self.search_candidates(
                activity, visit_day, effective_budget,
                current_location[0], current_location[1],
                categories=categories, guessed_category=guessed,
            )

            if candidates.empty:
                continue

            all_candidate_similarities.extend(candidates["similarity"].tolist())

            candidates = candidates[~candidates["Place_Name"].isin(visited_names)]
            if candidates.empty:
                continue

            # --- Pemilihan sekunder di dalam shortlist relevan ---
            # Rating, review, dan jarak baru berperan setelah Top-10 similarity
            # terbentuk. Kandidat dengan secondary_score tertinggi dicoba dulu.
            # Cek jam operasional, sisa waktu perjalanan, dan budget tersisa;
            # coba maksimal MAX_GREEDY_RETRY kandidat sebelum menyerah.
            shortlist = candidates.sort_values("secondary_score", ascending=False)

            picked_info = None
            attempts = 0
            for _, row in shortlist.iterrows():
                if attempts >= MAX_GREEDY_RETRY:
                    break
                attempts += 1

                distance = row["distance"]
                travel_min = self.estimate_travel_time(distance)
                arrive = current_time + timedelta(minutes=travel_min)

                try:
                    open_time = datetime.strptime(str(row["Open_time"]), "%H:%M:%S")
                    close_time = datetime.strptime(str(row["Close_time"]), "%H:%M:%S")
                except Exception:
                    continue

                if arrive < open_time:
                    arrive = open_time

                duration = int(row["estimated_duration_min"])
                leave = arrive + timedelta(minutes=duration)

                # Jam Operasional & Sisa Waktu Perjalanan
                if leave > close_time or leave > limit_time:
                    continue

                # Budget Tersisa
                cost = int(row["price_value"])
                if used_budget + cost > budget:
                    continue

                picked_info = {
                    "activity": activity,
                    "row": row,
                    "distance": distance,
                    "travel_min": travel_min,
                    "arrive": arrive,
                    "leave": leave,
                    "cost": cost,
                }
                break

            if picked_info is None:
                if DEBUG_SCORES:
                    print(f"[DEBUG PILIH] '{activity}' -> GAGAL setelah {attempts}x percobaan (jam/budget tidak muat)\n")
                # Tandai Gagal Mencari untuk aktivitas ini -> lanjut ke aktivitas
                # berikutnya (bukan menghentikan seluruh itinerary).
                continue

            if DEBUG_SCORES:
                r = picked_info["row"]
                print(
                    f"[DEBUG PILIH] '{activity}' -> TERPILIH: {r['Place_Name']} "
                    f"(similarity={r['similarity']:.3f}, sekunder={r['secondary_score']:.3f}, "
                    f"jarak={picked_info['distance']:.2f}km, "
                    f"percobaan ke-{attempts})\n"
                )

            # --- Sequential Time Scheduling ---
            used_budget += picked_info["cost"]
            current_time = picked_info["leave"]
            current_location = (
                float(picked_info["row"]["Lat"]),
                float(picked_info["row"]["Long"]),
            )
            visited_names.add(picked_info["row"]["Place_Name"])
            itinerary.append(picked_info)

        return itinerary, used_budget, current_time, all_candidate_similarities

    def build_itinerary_auto(self, forced_categories, visit_day, start_time, end_time,
                              budget, start_lat, start_lon):
        """Dipakai kalau user tidak memberi preferensi teks spesifik (mis. 'terserah',
        'bebas', 'rekomendasikan saja'). Tidak ada preferensi untuk dicocokkan
        secara semantik, jadi ranking di sini murni dari rating & jumlah review
        (lihat search_candidates_popularity) — bukan similarity teks.

        Kandidat disusun round-robin lintas kategori yang dipilih user (rangking-1
        tiap kategori dulu, baru rangking-2, dst) supaya tiap kategori punya
        kesempatan tampil di itinerary duluan, sebelum satu kategori "memonopoli"
        semua slot. Tetap tunduk ke batas jam & budget — kalau kandidat terbaik
        suatu kategori tidak muat jadwal/budget, kategori itu bisa saja tidak
        kebagian slot sama sekali.
        """

        categories = forced_categories or list(ALLOWED_CATEGORIES)

        current_time = datetime.strptime(start_time, "%H:%M")
        limit_time = datetime.strptime(end_time, "%H:%M")
        current_location = (start_lat, start_lon)
        used_budget = 0
        itinerary = []
        visited_names = set()
        all_candidate_similarities = []

        category_pools = {}
        for cat in categories:
            cand = self.search_candidates_popularity(visit_day, budget, [cat], top_n=15)
            if not cand.empty:
                category_pools[cat] = cand.reset_index(drop=True)
                all_candidate_similarities.extend(cand["similarity"].tolist())

        if not category_pools:
            return [], 0, current_time, []

        # Round-robin: rangking-1 tiap kategori dulu, baru rangking-2, dst.
        ordered_candidates = []
        max_len = max(len(pool) for pool in category_pools.values())
        for rank in range(max_len):
            for cat in categories:
                pool = category_pools.get(cat)
                if pool is not None and rank < len(pool):
                    ordered_candidates.append(pool.iloc[rank])

        for row in ordered_candidates:

            if row["Place_Name"] in visited_names:
                continue

            remaining_budget = budget - used_budget
            cost = int(row["price_value"])
            if cost > remaining_budget:
                continue

            distance = self.haversine_distance(
                current_location[0], current_location[1], float(row["Lat"]), float(row["Long"])
            )
            travel_min = self.estimate_travel_time(distance)
            arrive = current_time + timedelta(minutes=travel_min)

            try:
                open_time = datetime.strptime(str(row["Open_time"]), "%H:%M:%S")
                close_time = datetime.strptime(str(row["Close_time"]), "%H:%M:%S")
            except Exception:
                continue

            if arrive < open_time:
                arrive = open_time

            duration = int(row["estimated_duration_min"])
            leave = arrive + timedelta(minutes=duration)

            if leave > close_time or leave > limit_time:
                continue

            used_budget += cost
            current_time = leave
            current_location = (float(row["Lat"]), float(row["Long"]))
            visited_names.add(row["Place_Name"])

            itinerary.append({
                "activity": f"Rekomendasi {row['Category']}",
                "row": row,
                "distance": distance,
                "travel_min": travel_min,
                "arrive": arrive,
                "leave": leave,
                "cost": cost,
            })

            if current_time >= limit_time:
                break

        return itinerary, used_budget, current_time, all_candidate_similarities

    @staticmethod
    def evaluate(itinerary, candidate_similarities, start_time, end_time, budget, used_budget):

        avg_candidate_similarity = (
            sum(candidate_similarities) / len(candidate_similarities)
            if candidate_similarities else 0
        )

        itinerary_similarities = [item["row"]["similarity"] for item in itinerary]
        avg_itinerary_similarity = (
            sum(itinerary_similarities) / len(itinerary_similarities)
            if itinerary_similarities else 0
        )

        if avg_itinerary_similarity >= 0.70:
            quality = "Sangat Baik"
        elif avg_itinerary_similarity >= 0.50:
            quality = "Baik"
        elif avg_itinerary_similarity >= 0.30:
            quality = "Cukup"
        else:
            quality = "Perlu Perbaikan Dataset"

        total_distance = sum(item["distance"] for item in itinerary)
        total_visit_time = sum(int(item["row"]["estimated_duration_min"]) for item in itinerary)

        available_minutes = (
            datetime.strptime(end_time, "%H:%M") - datetime.strptime(start_time, "%H:%M")
        ).total_seconds() / 60
        time_utilization = (total_visit_time / available_minutes * 100) if available_minutes > 0 else 0

        route_efficiency = total_distance / len(itinerary) if itinerary else 0

        return {
            "avg_candidate_similarity": avg_candidate_similarity,
            "avg_itinerary_similarity": avg_itinerary_similarity,
            "quality": quality,
            "total_distance": total_distance,
            "route_efficiency": route_efficiency,
            "time_utilization": time_utilization,
        }

    # ---------------------------------------------------
    # HELPER TAMPILAN (dipakai oleh Flask/template)
    # ---------------------------------------------------

    @staticmethod
    def _pick_icon(category):
        return CATEGORY_ICON.get(str(category).strip().lower(), "icon-wisata.png")

    @staticmethod
    def _pick_photo(place_id):
        """Cari file foto di static/static_1/ dengan nama = Place_ID, coba
        beberapa ekstensi umum (jpg/jpeg/png/webp) karena tiap tempat bisa
        punya ekstensi berbeda. Return None kalau tidak ketemu, supaya
        template bisa fallback ke ikon placeholder seperti biasa.
        """
        try:
            place_id_str = str(int(place_id))
        except (TypeError, ValueError):
            place_id_str = str(place_id).strip()

        for ext in PHOTO_EXTENSIONS:
            filename = f"{place_id_str}{ext}"
            if os.path.exists(os.path.join(STATIC_PHOTO_DIR, filename)):
                return filename
        return None

    @staticmethod
    def _clean_maps_url(value):
        """Rapikan nilai Google_Maps_URL: None/NaN/string kosong -> None,
        supaya template bisa cukup cek `{% if p.maps_url %}` tanpa perlu
        tahu detail representasi NaN dari pandas.
        """
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text == "" or text.lower() in ("nan", "none"):
            return None
        return text

    def _serialize_itinerary(self, itinerary):
        places = []
        for item in itinerary:
            row = item["row"]
            places.append({
                "activity": item["activity"],
                "name": row["Place_Name"],
                "category": row["Category"],
                "open_time": str(row["Open_time"])[:5],
                "close_time": str(row["Close_time"])[:5],
                "duration_min": int(row["estimated_duration_min"]),
                "cost": item["cost"],
                "rating": float(row["Rating"]),
                "review_count": int(row["Review_count"]),
                "similarity": float(row["similarity"]),
                "distance_km": round(item["distance"], 2),
                "arrive": item["arrive"].strftime("%H:%M"),
                "leave": item["leave"].strftime("%H:%M"),
                "lat": float(row["Lat"]),
                "lng": float(row["Long"]),
                "icon": self._pick_icon(row["Category"]),
                "photo": self._pick_photo(row["Place_ID"]),
                "maps_url": self._clean_maps_url(row.get("Google_Maps_URL")),
            })
        return places

    @staticmethod
    def is_auto_order(narration):
        """True kalau narasi user pada dasarnya bilang 'terserah sistem aja'."""
        text = (narration or "").lower()
        return any(trigger in text for trigger in AUTO_ORDER_TRIGGERS)

    def plan_trip(self, narration, visit_day, start_time, end_time, budget, lat, lon,
                  forced_categories=None):
        """Entry point dipakai oleh Flask route. Mengembalikan dict siap-render."""

        auto_mode = self.is_auto_order(narration)

        if auto_mode:
            itinerary, used_budget, finish_time, candidate_similarities = self.build_itinerary_auto(
                forced_categories, visit_day, start_time, end_time, budget, lat, lon,
            )
        else:
            # Tanpa koma: seluruh narasi adalah satu preferensi yang boleh
            # menghasilkan beberapa tempat. Dengan koma: setiap klausa tetap
            # menjadi satu slot dan diproses persis sesuai urutan penulisan.
            narration_text = (narration or "").strip()
            fill_schedule = bool(narration_text) and "," not in narration_text
            raw_segments = (
                [narration_text]
                if fill_schedule
                else self.extract_intents(narration_text)
            )

            # Hanya pakai segmen yang memang menyebut kategori tempat (wisata/kuliner/
            # cafe/mall). Kalimat filler seperti "aku pengen liburan dari pagi sampe
            # sore" tidak punya kata kunci kategori sehingga tidak layak jadi 1 slot
            # itinerary tersendiri (dulu ini bikin similarity rendah / hasil ngasal).
            activities = [seg for seg in raw_segments if self.guess_category(seg) is not None]

            if not activities:
                if raw_segments:
                    # User SUDAH menulis narasi (1 klausa atau lebih), hanya saja tidak
                    # ada kata kunci kategori yang cocok di CATEGORY_KEYWORDS. Narasi
                    # aslinya WAJIB tetap dipakai sebagai query similarity — jangan
                    # dibuang ke template generik, karena itu menghapus preferensi
                    # spesifik user (mis. "yang romantis, ada view kota" jadi diganti
                    # "ngopi nongkrong santai cozy..." padahal user sudah cerita detail).
                    # Kategorinya sendiri tetap aman disaring oleh forced_categories
                    # di build_itinerary() lewat parameter categories.
                    activities = raw_segments
                elif forced_categories:
                    # Narasi benar-benar kosong, tapi user sudah pilih kategori
                    # lewat filter -> pakai template generik sebagai aktivitas.
                    activities = [AUTO_QUERY_BY_CATEGORY.get(c, c) for c in forced_categories]
                else:
                    # Narasi generik total (atau kosong) -> itinerary seimbang lintas kategori.
                    activities = list(DEFAULT_FALLBACK_ACTIVITIES)

            if fill_schedule and len(activities) == 1:
                # Jalankan query yang sama kembali dari posisi terbaru. visited_names
                # mencegah duplikasi; batas waktu, budget, dan Top-10 membatasi jumlah
                # tempat yang benar-benar masuk itinerary.
                activities = activities * TOP_N_PER_CATEGORY

            itinerary, used_budget, finish_time, candidate_similarities = self.build_itinerary(
                activities, visit_day, start_time, end_time, budget, lat, lon,
                forced_categories=forced_categories,
            )

        metrics = self.evaluate(
            itinerary, candidate_similarities, start_time, end_time, budget, used_budget
        )

        places = self._serialize_itinerary(itinerary)
        ranked = sorted(places, key=lambda p: p["similarity"], reverse=True)

        available_minutes = (
            datetime.strptime(end_time, "%H:%M") - datetime.strptime(start_time, "%H:%M")
        ).total_seconds() / 60

        budget_ok = used_budget <= budget
        time_ok = finish_time <= datetime.strptime(end_time, "%H:%M")

        return {
            "places": places,
            "top_recommendations": ranked[:5],
            "used_budget": used_budget,
            "budget": budget,
            "remaining_budget": budget - used_budget,
            "finish_time": finish_time.strftime("%H:%M"),
            "total_distance": round(metrics["total_distance"], 1),
            "total_duration_hours": round(available_minutes / 60, 1),
            "place_count": len(places),
            "quality": metrics["quality"],
            "avg_similarity": round(metrics["avg_itinerary_similarity"], 3),
            "budget_status": "Sesuai Budget" if budget_ok else "Melebihi Budget",
            "time_status": "Semua Tempat Masih Buka" if time_ok else "Ada Jadwal Mepet",
            "visit_day": visit_day,
            "start_time": start_time,
            "end_time": end_time,
            "order_mode": "auto" if auto_mode else "manual",
        }
