import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Train/Test Split
# ============================================================

def split_data(X, y):
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# TF-IDF Feature Engineering
# ============================================================

def create_tfidf_features(X_train, X_test):
    """
    Convert text into TF-IDF numerical features.
    """

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    return (
        X_train_tfidf,
        X_test_tfidf,
        vectorizer
    )


# ============================================================
# Logistic Regression
# ============================================================

def train_logistic_regression(X_train, y_train):

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(
        f"Logistic Regression trained "
        f"in {training_time:.4f} seconds"
    )

    return model


# ============================================================
# Random Forest
# ============================================================

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(
        f"Random Forest trained "
        f"in {training_time:.4f} seconds"
    )

    return model


# ============================================================
# XGBoost
# ============================================================

def train_xgboost(X_train, y_train):

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(
        f"XGBoost trained "
        f"in {training_time:.4f} seconds"
    )

    return model


# ============================================================
# Train All Models
# ============================================================

def train_all_models(X_train, y_train):
    """
    Train all three classification models.
    """

    models = {}

    print("\nTraining Logistic Regression...")
    models["logistic_regression"] = (
        train_logistic_regression(
            X_train,
            y_train
        )
    )

    print("\nTraining Random Forest...")
    models["random_forest"] = (
        train_random_forest(
            X_train,
            y_train
        )
    )

    print("\nTraining XGBoost...")
    models["xgboost"] = (
        train_xgboost(
            X_train,
            y_train
        )
    )

    return models


# ============================================================
# Complete Training Preparation
# ============================================================

def prepare_training_data(X, y):
    """
    Complete training-data preparation:

    1. Train/test split
    2. TF-IDF transformation
    """

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    (
        X_train_tfidf,
        X_test_tfidf,
        vectorizer
    ) = create_tfidf_features(
        X_train,
        X_test
    )

    print("\nTraining data shape:", X_train_tfidf.shape)
    print("Testing data shape :", X_test_tfidf.shape)

    return (
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        vectorizer
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    from src.ingestion.s3_loader import (
        load_data_from_s3
    )

    from src.preprocessing.preprocess import (
        preprocess_data
    )

    # --------------------------------------------------------
    # Load data from S3
    # --------------------------------------------------------

    df = load_data_from_s3()

    # --------------------------------------------------------
    # Preprocess data
    # --------------------------------------------------------

    X, y = preprocess_data(df)

    # --------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------

    (
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        vectorizer
    ) = prepare_training_data(X, y)

    # --------------------------------------------------------
    # Train all models
    # --------------------------------------------------------

    models = train_all_models(
        X_train_tfidf,
        y_train
    )

    print("\nAll models trained successfully.")

    print("\nModels:")
    for model_name in models:
        print(f"- {model_name}")