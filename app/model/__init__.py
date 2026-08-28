import os
import joblib
import boto3


# ============================================================
# Configuration
# ============================================================

AWS_REGION = os.getenv(
    "AWS_REGION",
    "ap-south-1"
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

LOCAL_MODEL_PATH = "/tmp/best_model.pkl"

LOCAL_VECTORIZER_PATH = "/tmp/tfidf_vectorizer.pkl"


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

    if not os.path.exists(LOCAL_MODEL_PATH):
        download_model()

    return joblib.load(
        LOCAL_MODEL_PATH
    )


# ============================================================
# Load Vectorizer
# ============================================================

def load_vectorizer():

    if not os.path.exists(LOCAL_VECTORIZER_PATH):
        download_vectorizer()

    return joblib.load(
        LOCAL_VECTORIZER_PATH
    )