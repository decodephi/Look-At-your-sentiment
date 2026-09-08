import os
from pathlib import Path
import joblib
import boto3


# ============================================================
# Configuration
# ============================================================

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1"
)

S3_BUCKET_NAME = os.getenv(
    "S3_BUCKET_NAME",
    "sentiment-mlops-bucket-2026"
)

MODEL_S3_KEY = os.getenv(
    "MODEL_S3_KEY",
    "models/best_model.pkl"
)

VECTORIZER_S3_KEY = os.getenv(
    "VECTORIZER_S3_KEY",
    "models/tfidf_vectorizer.pkl"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "best_model.pkl")))
LOCAL_VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", str(PROJECT_ROOT / "tfidf_vectorizer.pkl")))
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local").lower()


# ============================================================
# S3 Client
# ============================================================

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION
    )


# ============================================================
# Download Model
# ============================================================

def download_model():

    s3_client = get_s3_client()

    s3_client.download_file(
        S3_BUCKET_NAME,
        MODEL_S3_KEY,
        LOCAL_MODEL_PATH
    )


# ============================================================
# Helper Functions
# ============================================================

def _has_local_artifacts() -> bool:
    return (
        LOCAL_MODEL_PATH.exists()
        and LOCAL_VECTORIZER_PATH.exists()
    )


# ============================================================
# Artifact Setup
# ============================================================

def ensure_local_artifacts():
    """Prefer local packaged model artifacts and use S3 only when explicitly configured."""
    if _has_local_artifacts():
        return

    if MODEL_SOURCE != "s3":
        raise FileNotFoundError(
            f"Missing model artifacts: {LOCAL_MODEL_PATH} and {LOCAL_VECTORIZER_PATH}. "
            "Place the files in the project root or set MODEL_SOURCE=s3 and configure AWS credentials."
        )

    if not LOCAL_MODEL_PATH.exists():
        download_model()

    if not LOCAL_VECTORIZER_PATH.exists():
        download_vectorizer()


# ============================================================
# Download Vectorizer
# ============================================================

def download_vectorizer():

    s3_client = get_s3_client()

    s3_client.download_file(
        S3_BUCKET_NAME,
        VECTORIZER_S3_KEY,
        LOCAL_VECTORIZER_PATH
    )


# ============================================================
# Load Model
# ============================================================

def load_model():

    if not LOCAL_MODEL_PATH.exists():
        if MODEL_SOURCE != "s3":
            raise FileNotFoundError(
                f"Model artifact not found at {LOCAL_MODEL_PATH}. Run scripts/run_training.py or set MODEL_SOURCE=s3."
            )
        download_model()

    return joblib.load(
        LOCAL_MODEL_PATH
    )


# ============================================================
# Load Vectorizer
# ============================================================

def load_vectorizer():

    if not LOCAL_VECTORIZER_PATH.exists():
        if MODEL_SOURCE != "s3":
            raise FileNotFoundError(
                f"Vectorizer artifact not found at {LOCAL_VECTORIZER_PATH}. Run scripts/run_training.py or set MODEL_SOURCE=s3."
            )
        download_vectorizer()

    return joblib.load(
        LOCAL_VECTORIZER_PATH
    )