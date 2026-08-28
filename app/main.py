from fastapi import FastAPI

from app.api.routes import router


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Sentiment MLOps API",
    description="Sentiment classification service",
    version="1.0.0"
)


# ============================================================
# Routes
# ============================================================

app.include_router(
    router
)