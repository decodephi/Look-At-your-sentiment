import re
import string

import pandas as pd


# ============================================================
# Configuration
# ============================================================

TEXT_COLUMN = "review"
TARGET_COLUMN = "sentiment"


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean a single text review.
    """

    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# Dataset Validation
# ============================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate that the dataset contains the
    required columns.
    """

    required_columns = {
        TEXT_COLUMN,
        TARGET_COLUMN
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


# ============================================================
# Preprocessing Pipeline
# ============================================================

def preprocess_data(
    df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Perform preprocessing and return
    features (X) and target (y).
    """

    # Make a copy so the original DataFrame
    # is not modified.
    data = df.copy()

    # --------------------------------------------------------
    # 1. Validate dataset
    # --------------------------------------------------------

    validate_dataset(data)

    # --------------------------------------------------------
    # 2. Keep required columns
    # --------------------------------------------------------

    data = data[
        [TEXT_COLUMN, TARGET_COLUMN]
    ]

    # --------------------------------------------------------
    # 3. Remove missing values
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            TEXT_COLUMN,
            TARGET_COLUMN
        ]
    )

    # --------------------------------------------------------
    # 4. Remove duplicate rows
    # --------------------------------------------------------

    data = data.drop_duplicates()

    # --------------------------------------------------------
    # 5. Clean text
    # --------------------------------------------------------

    data[TEXT_COLUMN] = data[
        TEXT_COLUMN
    ].apply(clean_text)

    # --------------------------------------------------------
    # 6. Remove empty reviews
    # --------------------------------------------------------

    data = data[
        data[TEXT_COLUMN].str.strip() != ""
    ]

    # --------------------------------------------------------
    # 7. Normalize target labels
    # --------------------------------------------------------

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # 8. Convert sentiment to numerical labels
    # --------------------------------------------------------

    sentiment_mapping = {
        "negative": 0,
        "positive": 1
    }

    data["label"] = data[
        TARGET_COLUMN
    ].map(sentiment_mapping)

    # --------------------------------------------------------
    # 9. Validate target conversion
    # --------------------------------------------------------

    if data["label"].isnull().any():

        invalid_labels = (
            data.loc[
                data["label"].isnull(),
                TARGET_COLUMN
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown sentiment labels: {invalid_labels}"
        )

    # --------------------------------------------------------
    # 10. Create X and y
    # --------------------------------------------------------

    X = data[TEXT_COLUMN]

    y = data["label"].astype(int)

    print("Preprocessing completed successfully.")
    print(f"Final dataset size: {len(data)}")
    print(f"Positive samples: {(y == 1).sum()}")
    print(f"Negative samples: {(y == 0).sum()}")

    return X, y


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Import ingestion module
    from src.ingestion.s3_loader import (
        load_data_from_s3
    )

    # Load raw data from S3
    df = load_data_from_s3()

    # Preprocess
    X, y = preprocess_data(df)

    print("\nSample processed text:")
    print(X.head())

    print("\nSample labels:")
    print(y.head())