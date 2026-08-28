from app.model.loader import (
    load_model,
    load_vectorizer
)


# ============================================================
# Load Production Artifacts
# ============================================================

model = load_model()

vectorizer = load_vectorizer()


# ============================================================
# Sentiment Prediction
# ============================================================

def predict_sentiment(text: str):

    # Convert text into TF-IDF representation
    text_vector = vectorizer.transform(
        [text]
    )

    # Make prediction
    prediction = model.predict(
        text_vector
    )[0]

    # Convert numerical label to sentiment
    sentiment_mapping = {
        0: "negative",
        1: "positive"
    }

    sentiment = sentiment_mapping.get(
        int(prediction),
        "unknown"
    )

    # Calculate confidence
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            text_vector
        )[0]

        confidence = float(
            max(probabilities)
        )

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }