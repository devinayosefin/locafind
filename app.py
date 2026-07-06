import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_CACHE_DIR = os.path.join(BASE_DIR, ".runtime_cache")
os.makedirs(RUNTIME_CACHE_DIR, exist_ok=True)

# Hugging Face Docker Space tetap memakai tampilan Flask lama, tetapi backend
# similarity dibuat ringan supaya tidak perlu install model SBERT besar.
os.environ.setdefault("LOCAFIND_EMBEDDING_BACKEND", "tfidf")
os.environ.setdefault("TEMP", RUNTIME_CACHE_DIR)
os.environ.setdefault("TMP", RUNTIME_CACHE_DIR)

import engine

engine.DEBUG_SCORES = os.environ.get("LOCAFIND_DEBUG_SCORES", "0") == "1"

from flask_app import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
