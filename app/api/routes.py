from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    SentimentRequest,
    SentimentResponse
)

from app.model.predictor import (
    predict_sentiment
)


# ============================================================
# Router
# ============================================================

router = APIRouter()


# ============================================================
# Health Check
# ============================================================

@router.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# Sentiment Prediction
# ============================================================

@router.post(
    "/predict",
    response_model=SentimentResponse
)
def predict(request: SentimentRequest):

    text = request.text.strip()

    if not text:

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    result = predict_sentiment(
        text
    )

    return {
        "text": text,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"]
    }