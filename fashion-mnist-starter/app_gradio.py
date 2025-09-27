from __future__ import annotations
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import preprocess_image, LABELS

model = tf.keras.models.load_model("model/fashion_cnn.keras")

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
    demo.launch()
