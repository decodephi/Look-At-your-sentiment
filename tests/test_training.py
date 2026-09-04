import pandas as pd

from src.preprocessing.preprocess import preprocess_data
from src.training.train import prepare_training_data, train_logistic_regression


def test_training_preparation_and_logistic_model():
    dataframe = pd.DataFrame({"review": ["great film", "bad film"] * 4, "sentiment": ["positive", "negative"] * 4})
    features, labels = preprocess_data(dataframe)
    train_x, test_x, train_y, test_y, vectorizer = prepare_training_data(features, labels)
    model = train_logistic_regression(train_x, train_y)
    assert train_x.shape[1] == test_x.shape[1]
    assert len(train_y) + len(test_y) == len(labels)
    assert model.predict(vectorizer.transform(["great"]))[0] in (0, 1)