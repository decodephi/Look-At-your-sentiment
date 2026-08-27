import os
import boto3
from botocore.exceptions import ClientError


# ============================================================
# Configuration
# ============================================================

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

S3_BUCKET_NAME = os.getenv(
    "S3_BUCKET_NAME",
    "sentiment-mlops-bucket-2026"
)

LOCAL_FILE_PATH = os.getenv(
    "LOCAL_DATA_PATH",
    "../data/IMDB.csv"
)

S3_OBJECT_KEY = "raw/sentiment.csv"


# ============================================================
# Upload Function
# ============================================================

def upload_file_to_s3(local_file_path: str,bucket_name: str,s3_object_key: str) -> None:

    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION
    )

    try:
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            s3_object_key
        )

        print("Dataset uploaded successfully.")
        print(f"Bucket : {bucket_name}")
        print(f"Object : {s3_object_key}")

    except FileNotFoundError:
        print(f"File not found: {local_file_path}")

    except ClientError as error:
        print(f"AWS error occurred: {error}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    upload_file_to_s3(
        local_file_path=LOCAL_FILE_PATH,
        bucket_name=S3_BUCKET_NAME,
        s3_object_key=S3_OBJECT_KEY
    )