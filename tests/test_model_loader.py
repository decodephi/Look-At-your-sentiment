from pathlib import Path

from app.model import loader


def test_loader_uses_project_root_model_artifacts():
    project_root = Path(__file__).resolve().parents[1]

    assert Path(loader.LOCAL_MODEL_PATH) == project_root / "best_model.pkl"
    assert Path(loader.LOCAL_VECTORIZER_PATH) == project_root / "tfidf_vectorizer.pkl"
