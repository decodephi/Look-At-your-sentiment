import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):
    """
    Evaluate one trained classification model.
    """

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    start_time = time.time()

    predictions = model.predict(X_test)

    prediction_time = time.time() - start_time

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"{model_name}")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "negative",
                "positive"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=[
            "Negative",
            "Positive"
        ],
        yticklabels=[
            "Negative",
            "Positive"
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Prediction Time": prediction_time
    }


# ============================================================
# Evaluate All Models
# ============================================================

def evaluate_all_models(
    models,
    X_test,
    y_test
):
    """
    Evaluate every trained model.
    """

    results = []

    for model_name, model in models.items():

        result = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    return results_df


# ============================================================
# Select Best Model
# ============================================================

def select_best_model(
    models,
    results_df
):
    """
    Select the model with the highest F1 score.
    """

    best_index = results_df[
        "F1 Score"
    ].idxmax()

    best_model_name = results_df.loc[
        best_index,
        "Model"
    ]

    best_model = models[
        best_model_name
    ]

    return (
        best_model_name,
        best_model
    )


# ============================================================
# Model Comparison
# ============================================================

def compare_models(results_df):
    """
    Display and visualize model performance.
    """

    print("\n")
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Metric comparison
    # --------------------------------------------------------

    results_df.set_index(
        "Model"
    )[[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]].plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(
        "Sentiment Model Performance Comparison"
    )

    plt.ylabel("Score")

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    from src.ingestion.s3_loader import (
        load_data_from_s3
    )

    from src.preprocessing.preprocess import (
        preprocess_data
    )

    from src.training.train import (
        prepare_training_data,
        train_all_models
    )

    # --------------------------------------------------------
    # Ingestion
    # --------------------------------------------------------

    df = load_data_from_s3()

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    X, y = preprocess_data(df)

    # --------------------------------------------------------
    # Training data preparation
    # --------------------------------------------------------

    (
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        vectorizer
    ) = prepare_training_data(
        X,
        y
    )

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    models = train_all_models(
        X_train_tfidf,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate models
    # --------------------------------------------------------

    results_df = evaluate_all_models(
        models,
        X_test_tfidf,
        y_test
    )

    # --------------------------------------------------------
    # Compare models
    # --------------------------------------------------------

    compare_models(
        results_df
    )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    (
        best_model_name,
        best_model
    ) = select_best_model(
        models,
        results_df
    )

    print("\n")
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Model: {best_model_name}"
    )

    best_score = results_df.loc[
        results_df["Model"] == best_model_name,
        "F1 Score"
    ].iloc[0]

    print(
        f"F1 Score: {best_score:.4f}"
    )