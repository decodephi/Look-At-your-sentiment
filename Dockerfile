FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src/ ./src/
COPY app/ ./app/
COPY best_model.pkl tfidf_vectorizer.pkl ./

RUN python -m pip install --no-cache-dir -r requirements.txt \
	&& useradd --create-home --shell /usr/sbin/nologin appuser \
	&& chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]