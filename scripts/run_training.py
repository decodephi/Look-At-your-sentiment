"""Train and persist the best sentiment model from the local CSV dataset."""

from pathlib import Path

import pandas as pd

from src.artifacts.artifact_manager import save_model_artifacts
from src.evaluation.evaluate import evaluate_all_models, select_best_model
from src.preprocessing.preprocess import preprocess_data
from src.training.train import prepare_training_data, train_all_models


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataframe = pd.read_csv(root / "data" / "IMDB.csv")
    features, labels = preprocess_data(dataframe)
    train_x, test_x, train_y, test_y, vectorizer = prepare_training_data(features, labels)
    models = train_all_models(train_x, train_y)
    results = evaluate_all_models(models, test_x, test_y, show_plots=False)
    best_name, best_model = select_best_model(models, results)
    save_model_artifacts(best_model, vectorizer, root / "best_model.pkl", root / "tfidf_vectorizer.pkl")
    print(f"Saved {best_name} as the production model.")


if __name__ == "__main__":
    main()