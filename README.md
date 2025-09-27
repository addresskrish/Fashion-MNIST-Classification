# Fashion MNIST — Classification (Starter Repo)

A minimal, production-friendly starter for training a CNN on **Fashion MNIST** and serving it via **Gradio** or **FastAPI**.

## Quickstart

```bash
# 1) Create & activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train
python train.py

# 4a) Test a prediction (CLI)
python predict.py --image sample_ankle_boot.png

# 4b) Try the Gradio demo
python app_gradio.py
# then open the local URL it prints

# 4c) Run FastAPI (server)
uvicorn app_fastapi:app --reload --port 8000
# POST an image to http://localhost:8000/predict
```

## Repo layout

```
fashion-mnist-starter/
├── app_fastapi.py      # FastAPI server for inference
├── app_gradio.py       # Gradio demo UI
├── predict.py          # CLI inference
├── requirements.txt
├── train.py            # Train + evaluate + save model
├── utils.py            # Preprocessing & label map
├── sample_ankle_boot.png
└── README.md
```

## Notes

- **Dataset**: Loaded automatically via `tf.keras.datasets.fashion_mnist`.
- **Model**: Small CNN, ~100k params, ~91–93% test accuracy in a few epochs on CPU.
- **Saved model**: `./model/fashion_cnn.keras` (Keras v3 native format).
- **Serving**: Use **Gradio** for quick demo; **FastAPI** for integration.
- **Export**: Convert to TF Lite or ONNX later if you need mobile or Node runtimes.
