# CIFAR-10 Model API - Lab 3

## Description
A containerized FastAPI application that serves a trained CIFAR-10 image classification model. The PyTorch model from Lab 2 is exported to ONNX format and served via a POST /predict endpoint inside a Docker container.

## Requirements
- Python 3.12 or later
- `uv` package manager ([installation](https://docs.astral.sh/uv/))
- Docker

## Installation
```bash
git clone https://github.com/jazzuss/ml_ramverk_l3_Jasmin.git
cd ml_ramverk_l3_Jasmin
uv sync
```

## Usage

### Train the model and export to ONNX
```bash
uv run python main.py
uv run python export_onnx.py
```
This saves the trained model to `models/best_model.pth` and exports it to `models/model.onnx`.

### Run locally
```bash
uv run uvicorn api:app --reload
```

### Run with Docker
```bash
docker build -t ml-ramverk-l3 .
docker run -p 8000:8000 ml-ramverk-l3
```

### Test the API
```bash
uv run python -c "import requests; import numpy as np; dummy = np.random.randn(1, 3, 32, 32).tolist(); r = requests.post('http://127.0.0.1:8000/predict', json={'image': dummy}); print(r.json())"
```

Example response:
```json
{"label": "frog", "confidence": 0.46}
```

## Project Structure
```
ml_ramverk_l3_Jasmin/
├── api.py               # FastAPI endpoint (POST /predict)
├── model.py             # CNN architecture (nn.Module)
├── export_onnx.py       # Exports trained model to ONNX
├── main.py              # Trains the final model
├── train.py             # Training and evaluation logic
├── dataset.py           # CIFAR-10 Dataset and DataLoader
├── experiments.py       # Hyperparameter configurations
├── Dockerfile           # Container setup with uv
├── pyproject.toml       # Dependencies
├── uv.lock              # Locked versions
└── README.md
```

Note: `models/` and `data/` are not included in the repo. Run `uv run python main.py` and `uv run python export_onnx.py` to generate the model files.

## API
The API has one endpoint:

**POST /predict** accepts a JSON body with a 4D image array `[batch, channels, height, width]` and returns a class label with confidence score. The 10 CIFAR-10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Pull Requests
[PR #1: Add ONNX model export and FastAPI predict endpoint](https://github.com/jazzuss/ml_ramverk_l3_Jasmin/pull/1)
[PR #2: Add Dockerfile with uv](https://github.com/jazzuss/ml_ramverk_l3_Jasmin/pull/2)

## Reflection
This lab was about taking a trained model and making it usable outside of a training script. Exporting to ONNX was new to me, it was interesting to see how the model can run without PyTorch using onnxruntime. Building the FastAPI endpoint made the model accessible via HTTP, which is how models are typically used in production. Docker tied everything together so the whole application can run on any machine without worrying about dependencies.

## Reproducibility
1. Clone the repo and run `uv sync`
2. Run `uv run python main.py` to train the model
3. Run `uv run python export_onnx.py` to export to ONNX
4. Run `docker build -t ml-ramverk-l3 .` and `docker run -p 8000:8000 ml-ramverk-l3`