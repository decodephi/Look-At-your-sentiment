# Look-At-your-sentiment

# Look At Your Sentiment

An end-to-end sentiment classifier with a FastAPI inference service, React UI, reproducible training pipeline, MLflow tracking, and optional S3 artifact storage.

## Run locally

Use Python 3.10 or newer and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

The repository includes production artifacts at its root. To regenerate them from `data/IMDB.csv`:

```powershell
python scripts/run_training.py
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Start the UI in a second terminal:

```powershell
Set-Location frontend/sentiment-ui
npm install
npm run dev
```

Set `VITE_API_URL` when the API is not running at `http://127.0.0.1:8000`. The API exposes `GET /health`, `POST /predict`, and interactive docs at `/docs`.

## Tests and deployment

```powershell
pytest -q
docker build -t sentiment-api:latest .
kubectl apply -f infrastructure/kubernetes/deployment.yaml
```

For cloud artifact loading, set `MODEL_SOURCE=s3` and provide the AWS/S3 variables in `.env.example`. Never commit credentials; `credential.txt` should be removed or kept outside version control.

