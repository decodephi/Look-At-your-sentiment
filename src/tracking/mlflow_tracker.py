import os

import mlflow
import mlflow.sklearn


# ============================================================
# MLflow Configuration
# ============================================================

EXPERIMENT_NAME = "sentiment-classification"

# Local MLflow tracking using SQLite
# We will move this to an MLflow server + AWS later.
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)


# ============================================================
# Initialize MLflow
# ============================================================

def initialize_mlflow():
    """
    Configure MLflow tracking and experiment.
    """

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow experiment : {EXPERIMENT_NAME}")
    print(f"MLflow tracking   : {TRACKING_URI}")


# ============================================================
# Log Model Experiment
# ============================================================

def log_model_experiment(
    model,
    model_name,
    parameters,
    metrics
):
    """
    Log one model experiment into MLflow.
    """

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # Log model name
        # ----------------------------------------------------

        mlflow.set_tag(
            "model",
            model_name
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        for parameter_name, parameter_value in parameters.items():

            mlflow.log_param(
                parameter_name,
                parameter_value
            )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        for metric_name, metric_value in metrics.items():

            mlflow.log_metric(
                metric_name,
                float(metric_value)
            )

        # ----------------------------------------------------
        # Log model artifact
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print(
            f"MLflow run logged: {model_name}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    initialize_mlflow()

    print(
        "MLflow tracking initialized successfully."
    )