from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "data" / "IMDB.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "best_model.pkl"))
VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", ROOT_DIR / "tfidf_vectorizer.pkl"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))