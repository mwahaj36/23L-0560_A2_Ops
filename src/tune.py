"""Eight validation-only MobileNetV2 hyperparameter trials."""
from __future__ import annotations

import csv
from pathlib import Path

import keras_tuner as kt
import tensorflow as tf

from dataset import SEED, make_dataset
from train import ROOT, transfer


def build(hp):
    model, _ = transfer(224, hp.Choice("lr", [1e-3, 1e-4]),
                        hp.Choice("dropout", [0.2, 0.4]),
                        hp.Choice("unfrozen_layers", [0, 20]))
    return model


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)
    split_dir = ROOT / "reports" / "splits"
    train = make_dataset(split_dir / "train.csv", ROOT / "data_extracted", training=True)
    valid = make_dataset(split_dir / "val.csv", ROOT / "data_extracted")
    tuner = kt.RandomSearch(build, objective="val_auc", max_trials=8, seed=SEED,
                            directory="reports/tuner", project_name="visionguard", overwrite=True)
    tuner.search(train, validation_data=valid, epochs=3, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])
    output = Path("reports/tuner/visionguard_results.csv")
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial", "learning_rate", "dropout", "unfrozen_layers", "val_auc"])
        for trial in tuner.oracle.trials.values():
            hp = trial.hyperparameters.values
            writer.writerow([trial.trial_id, hp["lr"], hp["dropout"], hp["unfrozen_layers"], trial.score])
    print(output.read_text())
