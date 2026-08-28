from pydantic import BaseModel, Field


# ============================================================
# Prediction Request
# ============================================================

class SentimentRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify"
    )


# ============================================================
# Prediction Response
# ============================================================

class SentimentResponse(BaseModel):

    text: str

    sentiment: str

    confidence: float | None