from __future__ import annotations
import numpy as np
from PIL import Image

LABELS = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to normalized (1,28,28,1) float32 grayscale tensor."""
    # convert to grayscale, resize to 28x28
    img = img.convert("L").resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    # invert if background is white (optional heuristic)
    if arr.mean() > 0.7:
        arr = 1.0 - arr
    arr = arr.reshape(1, 28, 28, 1)
    return arr
