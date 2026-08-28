import os
import joblib
import boto3

from botocore.exceptions import ClientError


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

MODEL_S3_KEY = "models/best_model.pkl"

VECTORIZER_S3_KEY = "models/tfidf_vectorizer.pkl"


# ============================================================
# Save Artifacts Locally
# ============================================================

def save_model_artifacts(
    model,
    vectorizer,
    model_path="best_model.pkl",
    vectorizer_path="tfidf_vectorizer.pkl"
):
    """
    Save the trained model and TF-IDF vectorizer locally.
    """

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        vectorizer,
        vectorizer_path
    )

    print("Model artifacts saved locally.")

    print(f"Model     : {model_path}")
    print(f"Vectorizer: {vectorizer_path}")


# ============================================================
# Upload Artifacts to S3
# ============================================================

def upload_artifacts_to_s3(
    model_path="best_model.pkl",
    vectorizer_path="tfidf_vectorizer.pkl"
):
    """
    Upload model artifacts to S3.
    """

    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    try:

        # ----------------------------------------------------
        # Upload model
        # ----------------------------------------------------

        s3_client.upload_file(
            model_path,
            S3_BUCKET_NAME,
            MODEL_S3_KEY
        )

        print(
            f"Model uploaded to "
            f"s3://{S3_BUCKET_NAME}/{MODEL_S3_KEY}"
        )

        # ----------------------------------------------------
        # Upload vectorizer
        # ----------------------------------------------------

        s3_client.upload_file(
            vectorizer_path,
            S3_BUCKET_NAME,
            VECTORIZER_S3_KEY
        )

        print(
            f"Vectorizer uploaded to "
            f"s3://{S3_BUCKET_NAME}/{VECTORIZER_S3_KEY}"
        )

    except ClientError as error:

        print(
            f"Failed to upload artifacts: {error}"
        )

        raise


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "Artifact manager is ready."
    )