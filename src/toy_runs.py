"""Small Fashion-MNIST labs for MLflow and Keras Tuner."""
from __future__ import annotations

import argparse
import os

import keras_tuner as kt
import mlflow
import tensorflow as tf

from dataset import SEED


def data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return (x_train[:6000, ..., None] / 255.0, y_train[:6000],
            x_test[:1000, ..., None] / 255.0, y_test[:1000])


def model(lr, units):
    net = tf.keras.Sequential([tf.keras.layers.Input((28, 28, 1)), tf.keras.layers.Flatten(),
                               tf.keras.layers.Dense(units, activation="relu"),
                               tf.keras.layers.Dense(10, activation="softmax")])
    net.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    return net


def mlflow_lab():
    x_train, y_train, x_test, y_test = data()
    mlflow.set_experiment("23L-0560_A2_Ops_toy")
    for lr in (0.001, 0.0001):
        tf.keras.utils.set_random_seed(SEED)
        net = model(lr, 64)
        with mlflow.start_run(run_name=f"toy_lr_{lr}"):
            mlflow.log_params({"learning_rate": lr, "units": 64, "seed": SEED})
            history = net.fit(x_train, y_train, validation_split=0.2, epochs=3,
                              batch_size=64, verbose=0)
            for epoch in range(3):
                for key, values in history.history.items():
                    mlflow.log_metric(key, values[epoch], step=epoch)
            loss, accuracy = net.evaluate(x_test, y_test, verbose=0)
            mlflow.log_metrics({"test_loss": loss, "test_accuracy": accuracy})
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(history.history["accuracy"], label="train")
            ax.plot(history.history["val_accuracy"], label="validation")
            ax.legend()
            fig.savefig("toy_curve.png")
            plt.close(fig)
            mlflow.log_artifact("toy_curve.png")
            mlflow.keras.log_model(net, name="model")
            print({"lr": lr, "test_accuracy": accuracy})


def tuner_lab():
    x_train, y_train, _, _ = data()
    tuner = kt.RandomSearch(lambda hp: model(hp.Choice("lr", [0.001, 0.0001]),
                                             hp.Choice("units", [32, 64, 128])),
                            objective="val_accuracy", max_trials=4, seed=SEED,
                            directory="reports/tuner", project_name="fashion_mnist", overwrite=True)
    tuner.search(x_train, y_train, validation_split=0.2, epochs=2, batch_size=64, verbose=0)
    print("Toy best:", tuner.get_best_hyperparameters(1)[0].values)
    for trial in tuner.oracle.trials.values():
        print(trial.trial_id, trial.hyperparameters.values, trial.score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["mlflow", "tuner"])
    args = parser.parse_args()
    os.makedirs("reports/tuner", exist_ok=True)
    (mlflow_lab if args.command == "mlflow" else tuner_lab)()
