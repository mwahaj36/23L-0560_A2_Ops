"""Train, compare, evaluate, and log VisionGuard models."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from dataset import SEED, create_splits, make_dataset, read_split

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def metrics():
    return [tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")]


def baseline(image_size: int, lr: float, dropout: float):
    layers = tf.keras.layers
    image = layers.Input((image_size, image_size, 3))
    x = image
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(image, output, name="baseline_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=metrics())
    return model


def transfer(image_size: int, lr: float, dropout: float, unfrozen: int = 0):
    layers = tf.keras.layers
    image = layers.Input((image_size, image_size, 3))
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")
    backbone.trainable = bool(unfrozen)
    if unfrozen:
        for layer in backbone.layers[:-unfrozen]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    x = backbone(image, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(image, output, name="mobilenet_v2")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=metrics())
    return model, backbone


def unfreeze_top(backbone, count: int):
    backbone.trainable = True
    for layer in backbone.layers[:-count]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for key, value in (logs or {}).items():
            mlflow.log_metric(key, float(value), step=epoch)


def callbacks(path: Path, manual: bool):
    result = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                               restore_best_weights=True),
              tf.keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", save_best_only=True),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)]
    if manual:
        result.append(EpochLogger())
    return result


def probabilities(model, dataset):
    return model.predict(dataset, verbose=0).ravel()


def score(y, probs, threshold):
    predicted = (probs >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, predicted), "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0), "f1": f1_score(y, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y, probs)}


def choose_threshold(y, probs):
    # False accusations of real artwork are costly: maximize F1 with precision >= 0.90.
    candidates = np.arange(0.05, 0.951, 0.01)
    eligible = [(score(y, probs, t)["f1"], t) for t in candidates
                if score(y, probs, t)["precision"] >= 0.90]
    return float(max(eligible)[1]) if eligible else 0.5


def plot_history(history, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, field in zip(axes, ("loss", "accuracy")):
        axis.plot(history.history[field], label="train")
        axis.plot(history.history["val_" + field], label="validation")
        best = int(np.argmin(history.history["val_loss"]))
        axis.axvline(best, color="black", linestyle="--", label=f"best epoch {best + 1}")
        axis.set(xlabel="epoch", ylabel=field)
        axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_and_plot(model, paths, image_size, prep, run_dir):
    validation = make_dataset(paths["val"], ROOT / "data_extracted", image_size, training=False, preprocessing=prep)
    test = make_dataset(paths["test"], ROOT / "data_extracted", image_size, training=False, preprocessing=prep)
    y_val = read_split(paths["val"], ROOT / "data_extracted")[1]
    y_test = read_split(paths["test"], ROOT / "data_extracted")[1]
    val_probs = probabilities(model, validation)
    threshold = choose_threshold(y_val, val_probs)
    test_probs = probabilities(model, test)
    results = score(y_test, test_probs, threshold)
    results["threshold"] = threshold
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_probs >= threshold), display_labels=["Real", "AI"]).plot(ax=ax)
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC {results['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.legend()
    fig.savefig(run_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    test_paths = read_split(paths["test"], ROOT / "data_extracted")[0]
    with (run_dir / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "actual", "probability_ai", "predicted"])
        writer.writerows(zip(test_paths, y_test.astype(int), test_probs, (test_probs >= threshold).astype(int)))
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def run(name: str, kind: str, lr: float, dropout: float, image_size: int,
        augmentation: bool, epochs: int, unfrozen: int = 0, autolog: bool = False,
        final_test: bool = False):
    set_seed()
    paths = {name: ROOT / "reports" / "splits" / f"{name}.csv" for name in ("train", "val", "test")}
    if not all(path.exists() for path in paths.values()):
        paths = create_splits(ROOT / "reports" / "figures" / "audit.csv", ROOT / "reports" / "splits")
    run_dir = ROOT / "reports" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    prep = "unit" if kind == "baseline" else "mobilenet"
    train_data = make_dataset(paths["train"], ROOT / "data_extracted", image_size,
                              training=True, augmentation=augmentation, preprocessing=prep)
    valid_data = make_dataset(paths["val"], ROOT / "data_extracted", image_size, preprocessing=prep)
    model, backbone = (baseline(image_size, lr, dropout), None) if kind == "baseline" else transfer(image_size, lr, dropout, 0)
    model.summary(print_fn=lambda line: print(line))
    model_path = run_dir / "best.keras"
    if autolog:
        mlflow.tensorflow.autolog(log_models=False)
    with mlflow.start_run(run_name=name) as active:
        mlflow.log_params({"model": kind, "lr": lr, "dropout": dropout, "image_size": image_size,
                           "augmentation": augmentation, "unfrozen_layers": unfrozen, "seed": SEED})
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        mlflow.set_tags({"git_commit": commit or "uncommitted", "dvc_version": "data_extracted.dvc", "seed": str(SEED)})
        head_path = run_dir / "head_best.keras" if unfrozen else model_path
        history = model.fit(train_data, validation_data=valid_data, epochs=epochs,
                            callbacks=callbacks(head_path, manual=not autolog), verbose=2)
        plot_history(history, run_dir / ("training_curves_head.png" if unfrozen else "training_curves.png"))
        mlflow.log_metric("head_best_epoch", int(np.argmin(history.history["val_loss"])) + 1)
        if unfrozen:
            model = tf.keras.models.load_model(head_path)
            backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
            unfreeze_top(backbone, unfrozen)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=metrics())
            fine_path = run_dir / "fine_best.keras"
            fine = model.fit(train_data, validation_data=valid_data, epochs=epochs,
                             callbacks=callbacks(fine_path, manual=not autolog), verbose=2)
            plot_history(fine, run_dir / "training_curves_fine.png")
            mlflow.log_metric("fine_best_epoch", int(np.argmin(fine.history["val_loss"])) + 1)
            winner = fine_path if min(fine.history["val_loss"]) < min(history.history["val_loss"]) else head_path
            shutil.copy2(winner, model_path)
            mlflow.set_tag("selected_stage", "fine" if winner == fine_path else "head")
        model = tf.keras.models.load_model(model_path)
        val_prob = probabilities(model, valid_data)
        y_val = read_split(paths["val"], ROOT / "data_extracted")[1]
        threshold = choose_threshold(y_val, val_prob)
        validation_results = score(y_val, val_prob, threshold)
        validation_results["threshold"] = threshold
        mlflow.log_metrics({"val_final_" + key: value for key, value in validation_results.items()})
        results = evaluate_and_plot(model, paths, image_size, prep, run_dir) if final_test else validation_results
        if final_test:
            mlflow.log_metrics({"test_" + key: value for key, value in results.items()})
        mlflow.log_artifacts(str(run_dir), artifact_path="evidence")
        mlflow.log_artifacts(str(ROOT / "reports" / "splits"), artifact_path="splits")
        mlflow.log_artifact(str(ROOT / "requirements.txt"))
        mlflow.keras.log_model(model, name="model")
        print(json.dumps({"run_id": active.info.run_id, "name": name, **results}, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", choices=["baseline", "transfer"], default="transfer")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--unfrozen", type=int, default=0)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--autolog", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    run(args.name, args.kind, args.lr, args.dropout, args.image_size,
        not args.no_augmentation, args.epochs, args.unfrozen, args.autolog, args.final_test)
