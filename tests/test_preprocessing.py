import pandas as pd
import pytest

from src.preprocessing.preprocess import clean_text, preprocess_data


def test_clean_text_removes_markup_and_normalizes_case():
    assert clean_text("<b>Great</b> film!  https://example.com") == "great film"


def test_preprocess_data_returns_numeric_labels():
    features, labels = preprocess_data(pd.DataFrame({"review": ["Good", "Bad"], "sentiment": ["positive", "negative"]}))
    assert features.tolist() == ["good", "bad"]
    assert labels.tolist() == [1, 0]


def test_preprocess_data_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Unknown sentiment labels"):
        preprocess_data(pd.DataFrame({"review": ["Okay"], "sentiment": ["neutral"]}))