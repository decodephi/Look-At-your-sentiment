import mlflow

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

from src.evaluation.evaluate import (
    evaluate_all_models,
    compare_models,
    select_best_model
)

from src.tracking.mlflow_tracker import (
    initialize_mlflow,
    log_model_experiment
)

from src.artifacts.artifact_manager import (
    save_model_artifacts,
    upload_artifacts_to_s3
)

# ============================================================
# Pipeline Configuration
# ============================================================

PIPELINE_NAME = "sentiment-training-pipeline"


# ============================================================
# Main Training Pipeline
# ============================================================

def run_training_pipeline():

    print("\n" + "=" * 70)
    print(f"STARTING {PIPELINE_NAME}")
    print("=" * 70)

    # ========================================================
    # STEP 1 — Data Ingestion
    # ========================================================

    print("\n[1/6] Loading dataset from S3...")

    df = load_data_from_s3()

    # ========================================================
    # STEP 2 — Preprocessing
    # ========================================================

    print("\n[2/6] Preprocessing dataset...")

    X, y = preprocess_data(df)

    # ========================================================
    # STEP 3 — Train/Test Split + TF-IDF
    # ========================================================

    print("\n[3/6] Preparing training data...")

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

    # ========================================================
    # STEP 4 — Train Models
    # ========================================================

    print("\n[4/6] Training models...")

    models = train_all_models(
        X_train_tfidf,
        y_train
    )

    # ========================================================
    # STEP 5 — Evaluate Models
    # ========================================================

    print("\n[5/6] Evaluating models...")

    results_df = evaluate_all_models(
        models,
        X_test_tfidf,
        y_test
    )

    compare_models(
        results_df
    )

    # ========================================================
    # STEP 6 — MLflow Tracking
    # ========================================================

    print("\n[6/6] Logging experiments to MLflow...")

    initialize_mlflow()

    for model_name, model in models.items():

        # Get metrics for this model
        model_result = results_df[
            results_df["Model"] == model_name
        ].iloc[0]

        # Parameters
        parameters = model.get_params()

        # Metrics
        metrics = {
            "accuracy": model_result["Accuracy"],
            "precision": model_result["Precision"],
            "recall": model_result["Recall"],
            "f1_score": model_result["F1 Score"],
            "prediction_time": model_result[
                "Prediction Time"
            ]
        }

        log_model_experiment(
            model=model,
            model_name=model_name,
            parameters=parameters,
            metrics=metrics
        )

    # ========================================================
    # Select Best Model
    # ========================================================

    (
        best_model_name,
        best_model
    ) = select_best_model(
        models,
        results_df
    )
    
    # ========================================================
    # Save Best Model Artifacts
    # ========================================================

    print("\nSaving best model artifacts...")

    save_model_artifacts(
        model=best_model,
        vectorizer=vectorizer
    )
    
    # ========================================================
    # Upload Artifacts to S3
    # ========================================================

    print("\nUploading model artifacts to S3...")

    upload_artifacts_to_s3()
    
    #####################

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Model    : {best_model_name}"
    )

    best_score = results_df.loc[
        results_df["Model"] == best_model_name,
        "F1 Score"
    ].iloc[0]

    print(
        f"F1 Score : {best_score:.4f}"
    )

    print("\nTraining pipeline completed successfully.")

    return {
        "models": models,
        "results": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "vectorizer": vectorizer
    }


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    run_training_pipeline()