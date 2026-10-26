"""Shared image validation and model inference for local and cloud serving."""
from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}


def decode_image(filename: str, data: bytes):
    if "." not in filename or filename.rsplit(".", 1)[1].lower() not in EXTENSIONS:
        raise ValueError("Please upload a JPG, PNG, WebP, or GIF image.")
    if len(data) > MAX_BYTES:
        raise ValueError("The image must be 5 MB or smaller.")
    if not data:
        raise ValueError("Please choose an image to upload.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            dimensions = image.size
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("This file cannot be decoded as an image.") from error
    return rgb, dimensions


@lru_cache(maxsize=1)
def load_model():
    import tensorflow as tf
    path = Path(os.environ.get("MODEL_PATH", "models/champion.keras"))
    if not path.exists():
        raise FileNotFoundError(f"Model unavailable: {path}")
    return tf.keras.models.load_model(path)


def predict(image: Image.Image):
    import tensorflow as tf
    model = load_model()
    size = int(model.input_shape[1])
    # Training uses direct resize and MobileNetV2's [-1, 1] transform.
    array = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(array)[None, ...]
    probability_ai = float(model.predict(batch, verbose=0)[0][0])
    threshold = float(os.environ.get("THRESHOLD", "0.5"))
    label = "AI-generated" if probability_ai >= threshold else "Real"
    return {"label": label, "confidence": round(100 * (probability_ai if label == "AI-generated" else 1 - probability_ai), 2),
            "probability_ai": probability_ai, "threshold": threshold,
            "model_version": os.environ.get("MODEL_VERSION", "champion")}
