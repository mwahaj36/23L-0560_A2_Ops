"""Download the champion registry alias and classify one local image."""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import tensorflow as tf
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    client = MlflowClient()
    model_name = "23L-0560_A2_Ops"
    mv = client.get_model_version_by_alias(model_name, "champion")
    print(f"Resolved alias 'champion' -> Version {mv.version} (run_id: {mv.run_id})")

    local_cached = Path("models/champion.keras")
    if local_cached.exists():
        print(f"Loading champion model from local cache: {local_cached}")
        model = tf.keras.models.load_model(local_cached)
    else:
        print(f"Downloading champion artifact from DagsHub MLflow run {mv.run_id}...")
        local_path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="evidence/best.keras")
        model = tf.keras.models.load_model(local_path)

    size = int(model.input_shape[1])
    with Image.open(args.image) as image:
        array = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)
    probability_ai = float(model.predict(tf.keras.applications.mobilenet_v2.preprocess_input(array)[None, ...], verbose=0)[0][0])
    print(json.dumps({"image": str(args.image), "probability_ai": round(probability_ai, 4),
                      "label": "AI-generated" if probability_ai >= 0.5 else "Real"}, indent=2))
