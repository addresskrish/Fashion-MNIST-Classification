# app_gradio.py
from __future__ import annotations
import os
from pathlib import Path
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import preprocess_image, LABELS

# base path for robust loading on different systems and hosts
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "fashion_cnn.keras"

# try to load model and give a helpful error if missing
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"model not found at {MODEL_PATH}. "
                            "Make sure the saved model is in model/fashion_cnn.keras")

model = tf.keras.models.load_model(str(MODEL_PATH))

def predict(image: Image.Image):
    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {label: float(probs[i]) for i, label in enumerate(LABELS)}, LABELS[idx]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload clothing image (28x28 recommended)"),
    outputs=[gr.Label(num_top_classes=10, label="Probabilities"), gr.Textbox(label="Top-1")],
    title="Fashion MNIST Classifier",
    description="A simple CNN trained on Fashion MNIST. Tip: upload a 28x28 grayscale sketch or a sample from the dataset."
)

if __name__ == "__main__":
    # use $PORT if provided by host, else default 7860
    port = int(os.environ.get("PORT", 7860))
    # bind to 0.0.0.0 so external hosts can reach it
    demo.launch(server_name="0.0.0.0", server_port=port)