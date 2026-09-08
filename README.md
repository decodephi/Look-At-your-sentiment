# Look At Your Sentiment

An end-to-end sentiment analysis application for IMDB-style reviews. It combines a
scikit-learn model, a FastAPI inference service, a React/Vite interface, a reproducible
training pipeline, MLflow tracking, and optional S3 artifact storage.

## Features

- Predicts positive or negative sentiment from text.
- Returns the model confidence with every prediction.
- Provides a browser-based React interface.
- Exposes health checks and interactive OpenAPI documentation.
- Supports local model artifacts and cloud-backed S3 artifacts.
- Includes Docker and Kubernetes deployment manifests.

## Project structure

```text
app/                         FastAPI application and model inference code
data/                        IMDB training data
frontend/sentiment-ui/       React/Vite frontend
infrastructure/kubernetes/  Kubernetes Deployment and Service manifests
scripts/run_training.py      Train and save production model artifacts
src/                         Training, preprocessing, evaluation, and tracking code
tests/                       API, model, preprocessing, and training tests
```

## Prerequisites

- Python 3.10 or newer
- Node.js and npm for the frontend
- Docker for container deployment
- `kubectl` and a Kubernetes cluster for Kubernetes deployment

## Local setup

Create and activate a virtual environment, then install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

The repository contains the production artifacts `best_model.pkl` and
`tfidf_vectorizer.pkl`. Regenerate them from `data/IMDB.csv` when needed:

```powershell
python scripts/run_training.py
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs`
for the interactive Swagger UI.

In a second terminal, install and start the frontend:

```powershell
Set-Location frontend/sentiment-ui
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`. To use a different
API URL, set `VITE_API_URL` before starting or building the frontend:

```powershell
$env:VITE_API_URL = "http://127.0.0.1:8000"
npm run dev
```

## API

### Health check

```http
GET /health
```

Response:

```json
{"status":"healthy"}
```

### Predict sentiment

```http
POST /predict
Content-Type: application/json
```

Request:

```json
{"text":"The performances were excellent and the story was engaging."}
```

Example response:

```json
{
  "text": "The performances were excellent and the story was engaging.",
  "sentiment": "positive",
  "confidence": 0.98
}
```

Blank text is rejected with HTTP 400.

## Configuration

The API supports these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `best_model.pkl` | Path to the trained model |
| `VECTORIZER_PATH` | `tfidf_vectorizer.pkl` | Path to the TF-IDF vectorizer |
| `DATA_PATH` | `data/IMDB.csv` | Path to the training dataset |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |

Do not commit passwords, access keys, or tokens. Keep `credential.txt` outside version
control and use environment variables or a secret manager for deployment credentials.

## Testing and quality checks

Run the Python test suite:

```powershell
pytest -q
```

Build and lint the frontend:

```powershell
Set-Location frontend/sentiment-ui
npm run lint
npm run build
```

## Docker

Build and run the API image from the repository root:

```powershell
docker build -t sentiment-api:latest .
docker run --rm -p 8000:8000 sentiment-api:latest
```

Check `http://localhost:8000/health` after the container starts.

## Kubernetes

The manifest creates a two-replica API Deployment and a LoadBalancer Service. The
EKS cluster must be created separately and the worker nodes must be allowed to pull
from the ECR repository.

### Create an EKS cluster

Install and configure `eksctl`, then create the cluster in your AWS account:

```powershell
eksctl create cluster --name sentiment-eks --region us-east-1 --nodes 2
```

This creates the EKS control plane and managed node group. The command can take
several minutes and incurs AWS charges.

### Build and push to ECR manually

Set your AWS account and region, create the repository, then authenticate Docker:

```powershell
$env:AWS_REGION = "us-east-1"
$env:AWS_ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
$env:ECR_REPOSITORY = "sentiment-api"
aws ecr create-repository --repository-name $env:ECR_REPOSITORY --region $env:AWS_REGION
aws ecr get-login-password --region $env:AWS_REGION | docker login --username AWS --password-stdin "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com"
docker build -t "$env:ECR_REPOSITORY`:latest" .
docker tag "$env:ECR_REPOSITORY`:latest" "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/$env:ECR_REPOSITORY`:latest"
docker push "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/$env:ECR_REPOSITORY`:latest"
```

Configure `kubectl` for EKS and deploy the image:

```powershell
aws eks update-kubeconfig --name sentiment-eks --region $env:AWS_REGION
kubectl apply -f infrastructure/kubernetes/deployment.yaml
kubectl set image deployment/sentiment-api sentiment-api="$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/$env:ECR_REPOSITORY`:latest"
kubectl rollout status deployment/sentiment-api
kubectl get pods -l app=sentiment-api
kubectl get service sentiment-api
```

The deployment uses `/health` for readiness and liveness probes.

### GitHub Actions deployment

The **Build and deploy API to ECR and EKS** workflow performs the ECR push and EKS
rollout. Configure a GitHub Actions secret named `AWS_ROLE_ARN` for an IAM role
trusted by GitHub Actions through OIDC. The role needs permission to manage the
named ECR repository and update the target EKS cluster. Run the workflow manually
and provide the AWS region, EKS cluster name, and ECR repository name.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

