from __future__ import annotations
import argparse, pathlib
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import preprocess_image, LABELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--model", default="model/fashion_cnn.keras", help="Path to saved Keras model")
    args = parser.parse_args()

    # Load
    model = tf.keras.models.load_model(args.model)

    # Preprocess
    img = Image.open(args.image)
    x = preprocess_image(img)

    # Predict
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    print(f"Predicted: {LABELS[idx]}")
    print("Class probabilities:")
    for i, p in enumerate(probs):
        print(f"{LABELS[i]:>12}: {p:.3f}")

if __name__ == "__main__":
    main()
