"""Reproducible stratified splits and TensorFlow image input pipeline."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 560
LABELS = {"Real": 0, "AI": 1}


def create_splits(audit_csv: Path, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    with audit_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Group exact duplicates before splitting so identical pixels cannot leak.
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sha256"], []).append(row)
    groups = list(grouped.values())
    conflicting = [group for group in groups if len({row["label"] for row in group}) > 1]
    if conflicting:
        raise ValueError(f"{len(conflicting)} identical images have conflicting labels")
    group_labels = [group[0]["label"] for group in groups]
    train, rest = train_test_split(groups, test_size=0.30, random_state=SEED, stratify=group_labels)
    valid, test = train_test_split(rest, test_size=0.50, random_state=SEED,
                                   stratify=[group[0]["label"] for group in rest])
    paths = {}
    for name, subset in (("train", train), ("val", valid), ("test", test)):
        path = output / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label", "sha256"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in ("path", "label", "sha256")}
                             for group in subset for row in group)
        paths[name] = path
    return paths


def read_split(csv_path: Path, root: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return ([str(root / row["path"]) for row in rows],
            np.array([LABELS[row["label"]] for row in rows], dtype=np.float32))


def make_dataset(csv_path: Path, root: Path, image_size: int = 224,
                 batch_size: int = 32, training: bool = False,
                 augmentation: bool = True, preprocessing: str = "mobilenet"):
    import tensorflow as tf

    paths, labels = read_split(csv_path, root)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.08, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
    ])

    def decode(path):
        def pil_decode(value):
            with Image.open(value.numpy().decode()) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        image = tf.py_function(pil_decode, [path], Tout=tf.uint8)
        image.set_shape((None, None, 3))
        return tf.image.resize(image, (image_size, image_size))

    def prepare(path, label):
        image = decode(path)
        if training and augmentation:
            image = aug(image, training=True)
        if preprocessing == "mobilenet":
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        elif preprocessing == "unit":
            image = image / 255.0
        else:
            raise ValueError(preprocessing)
        return image, label

    return dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
