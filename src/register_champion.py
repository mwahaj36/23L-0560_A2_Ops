import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import tensorflow as tf

def register_champion(model_name="VisionGuard"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    runs = mlflow.search_runs(
        experiment_names=["VisionGuard_Classification"],
        order_by=["metrics.test_roc_auc DESC"]
    )
    
    if runs.empty:
        raise ValueError("No runs found in experiment.")

    best_run = runs.iloc[0]
    second_best_run = runs.iloc[1]

    print(f"[*] Best run ID: {best_run.run_id} (ROC-AUC: {best_run['metrics.test_roc_auc']:.4f})")
    print(f"[*] Second best run ID: {second_best_run.run_id} (ROC-AUC: {second_best_run['metrics.test_roc_auc']:.4f})")

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    v1 = client.create_model_version(
        name=model_name,
        source="models/baseline_best.keras",
        run_id=second_best_run.run_id
    )
    print(f"[OK] Registered Model Version: {v1.version} (Baseline)")

    v2 = client.create_model_version(
        name=model_name,
        source="models/transfer_best.keras",
        run_id=best_run.run_id
    )
    print(f"[OK] Registered Model Version: {v2.version} (Transfer Learning Champion)")

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=v2.version
    )
    print(f"[OK] Assigned alias 'champion' to Version: {v2.version}")

    champion_meta = client.get_model_version_by_alias(model_name, "champion")
    champion_source = champion_meta.source
    print(f"[*] Champion model path from alias: {champion_source}")

    shutil.copy(champion_source, "models/champion_model.keras")
    print(f"[OK] Stored active serving model at: models/champion_model.keras")

    champion_model = tf.keras.models.load_model("models/champion_model.keras")
    test_img = np.random.uniform(0, 1, (1, 128, 128, 3)).astype(np.float32)
    pred_prob = champion_model.predict(test_img, verbose=0)[0][0]
    label = "AI" if pred_prob >= 0.5 else "Real"
    
    print("\n" + "="*50)
    print("      CHAMPION MODEL INFERENCE TEST (Task 3.6)")
    print("="*50)
    print(f"  Model Version: {champion_meta.version}")
    print(f"  Model Source:  {champion_source}")
    print(f"  Prediction:    {label}")
    print(f"  Confidence:    {pred_prob:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    register_champion()
