from __future__ import annotations
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from utils import preprocess_image, LABELS

app = FastAPI(title="Fashion MNIST Classifier API")
model = tf.keras.models.load_model("model/fashion_cnn.keras")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(BytesIO(content))
    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return JSONResponse({
        "top1_index": idx,
        "top1_label": LABELS[idx],
        "probs": {LABELS[i]: float(p) for i, p in enumerate(probs)}
    })
