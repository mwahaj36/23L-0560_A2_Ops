"""Download the champion registry alias and classify one local image."""
import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    model = mlflow.keras.load_model("models:/23L-0560_A2_Ops/champion")
    size = int(model.input_shape[1])
    with Image.open(args.image) as image:
        array = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)
    probability_ai = float(model.predict(tf.keras.applications.mobilenet_v2.preprocess_input(array)[None, ...], verbose=0)[0][0])
    print(json.dumps({"image": str(args.image), "probability_ai": probability_ai,
                      "label": "AI-generated" if probability_ai >= 0.5 else "Real"}, indent=2))
