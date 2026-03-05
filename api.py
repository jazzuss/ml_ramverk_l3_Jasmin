import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

app = FastAPI()
session = ort.InferenceSession("models/model.onnx")


class PredictRequest(BaseModel):
    image: list[list[list[list[float]]]]  # [batch, channels, height, width]


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    input_array = np.array(request.image, dtype=np.float32)
    outputs = session.run(None, {"image": input_array})
    probabilities = np.exp(outputs[0]) / np.exp(outputs[0]).sum(axis=1, keepdims=True)
    class_idx = int(probabilities.argmax())
    confidence = float(probabilities[0][class_idx])

    return PredictResponse(label=CLASSES[class_idx], confidence=confidence)