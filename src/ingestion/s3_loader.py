import io
import os

import boto3
import pandas as pd
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

S3_OBJECT_KEY = os.getenv(
    "S3_OBJECT_KEY",
    "raw/sentiment.csv"
)


# ============================================================
# S3 Client
# ============================================================

def get_s3_client():
    """
    Create and return an AWS S3 client.
    """

    return boto3.client(
        "s3",
        region_name=AWS_REGION
    )


# ============================================================
# Load Dataset from S3
# ============================================================

def load_data_from_s3(
    bucket_name: str = S3_BUCKET_NAME,
    object_key: str = S3_OBJECT_KEY
) -> pd.DataFrame:
    """
    Download a CSV file from S3 and return it
    as a pandas DataFrame.
    """

    s3_client = get_s3_client()

    try:

        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=object_key
        )

        file_content = response["Body"].read()

        dataframe = pd.read_csv(
            io.BytesIO(file_content)
        )

        print("Dataset loaded successfully from S3.")
        print(f"Bucket : {bucket_name}")
        print(f"Object : {object_key}")
        print(f"Shape  : {dataframe.shape}")

        return dataframe

    except ClientError as error:

        print(
            f"Failed to load dataset from S3: {error}"
        )

        raise


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = load_data_from_s3()

    print("\nFirst 5 rows:")
    print(df.head())