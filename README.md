# Iris Neural Network – MLOps Pipeline

Neural network (scikit-learn MLP) on the Iris dataset, served with FastAPI, containerised with Docker,
tested and built by GitHub Actions, and deployed to Render from Git.

## Pipeline
`train (src/train.py)` -> `accuracy gate (>= 100% test)` -> `pytest` -> `Docker build` -> `push to GitHub` -> `Render auto-deploy`

## Run locally
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements-dev.txt
    python -m src.train
    python -m pytest -q
    uvicorn src.app:app --reload                            # http://127.0.0.1:8000/docs

## Docker
    docker build -t iris-mlops .
    docker run -p 8000:8000 iris-mlops

## Example request
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
