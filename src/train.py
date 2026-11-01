import os
import argparse
import random
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from dotenv import load_dotenv

from dataset import create_dataset_from_csv
from models import build_baseline_cnn, build_transfer_model
from evaluate import evaluate_model

load_dotenv()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "transfer"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--augment", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine_tune_at", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("VisionGuard_Classification")

    img_size = (args.img_size, args.img_size)
    train_ds = create_dataset_from_csv("data/splits/train.csv", img_size=img_size, batch_size=args.batch_size, is_training=True, augment=args.augment)
    val_ds = create_dataset_from_csv("data/splits/val.csv", img_size=img_size, batch_size=args.batch_size, is_training=False, augment=False)
    test_ds = create_dataset_from_csv("data/splits/test.csv", img_size=img_size, batch_size=args.batch_size, is_training=False, augment=False)

    with mlflow.start_run(run_name=f"{args.model_type}_lr_{args.lr}_aug_{args.augment}") as run:
        mlflow.log_params({
            "model_type": args.model_type,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "data_augmentation": args.augment,
            "seed": args.seed,
            "fine_tune_at": args.fine_tune_at if args.model_type == "transfer" else "N/A"
        })

        mlflow.set_tags({
            "git_commit": os.getenv("GIT_COMMIT", "local_dev"),
            "dvc_data_version": "v1.0",
            "model_family": "VisionGuard",
            "seed": str(args.seed)
        })

        if args.model_type == "baseline":
            model = build_baseline_cnn(input_shape=(args.img_size, args.img_size, 3))
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        else:
            model, base_model = build_transfer_model(
                input_shape=(args.img_size, args.img_size, 3),
                fine_tune_at=args.fine_tune_at
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )

        model.summary()

        os.makedirs("models", exist_ok=True)
        checkpoint_path = f"models/{args.model_type}_best.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks
        )

        for epoch in range(len(history.history["loss"])):
            mlflow.log_metrics({
                "train_loss": history.history["loss"][epoch],
                "train_accuracy": history.history["accuracy"][epoch],
                "train_precision": history.history["precision"][epoch],
                "train_recall": history.history["recall"][epoch],
                "train_auc": history.history["auc"][epoch],
                "val_loss": history.history["val_loss"][epoch],
                "val_accuracy": history.history["val_accuracy"][epoch],
                "val_precision": history.history["val_precision"][epoch],
                "val_recall": history.history["val_recall"][epoch],
                "val_auc": history.history["val_auc"][epoch]
            }, step=epoch + 1)

        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        mlflow.log_metric("best_epoch", best_epoch)
        print(f"[OK] Best Epoch based on val_loss: {best_epoch}")

        test_metrics, cm_path, roc_path, y_true, y_probs = evaluate_model(
            model, test_ds, output_prefix=f"{args.model_type}_{run.info.run_id[:6]}"
        )

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        print("\n" + "="*50)
        print(f"       TEST EVALUATION RESULTS ({args.model_type.upper()})")
        print("="*50)
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("="*50 + "\n")

        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(roc_path, artifact_path="evaluation_plots")
        mlflow.log_artifact("data/splits/train.csv", artifact_path="splits")
        mlflow.log_artifact("data/splits/val.csv", artifact_path="splits")
        mlflow.log_artifact("data/splits/test.csv", artifact_path="splits")

        model.save(checkpoint_path)
        mlflow.keras.log_model(model, artifact_path="model")
        print(f"[OK] Run {run.info.run_id} logged successfully to MLflow!")

if __name__ == "__main__":
    main()
