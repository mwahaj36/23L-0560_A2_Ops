import os
import tensorflow as tf
import numpy as np
import pandas as pd
from dataset import create_dataset_from_csv
from models import build_transfer_model

def run_toy_fashion_mnist():
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:1000].astype("float32") / 255.0
    y_train = y_train[:1000]
    x_val = x_val[:300].astype("float32") / 255.0
    y_val = y_val[:300]

    toy_configs = [
        {"units": 32, "lr": 1e-3},
        {"units": 64, "lr": 1e-3},
        {"units": 128, "lr": 1e-4}
    ]

    best_val_acc = 0.0
    best_config = None

    print("\n" + "="*55)
    print("    TASK 2.7: TOY TUNING EXPERIMENTS (Fashion-MNIST)")
    print("="*55)
    for i, cfg in enumerate(toy_configs, 1):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(cfg["units"], activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, verbose=0)
        val_acc = history.history["val_accuracy"][-1]
        print(f"Trial {i}: Units={cfg['units']}, LR={cfg['lr']} -> Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = cfg

    print(f"\nBest Toy Config: {best_config} with Val Accuracy: {best_val_acc:.4f}")
    print("="*55 + "\n")

def run_visionguard_tuning():
    train_ds = create_dataset_from_csv("data/splits/train.csv", batch_size=32, is_training=True, augment=True)
    val_ds = create_dataset_from_csv("data/splits/val.csv", batch_size=32, is_training=False, augment=False)

    search_space = [
        {"lr": 1e-3, "dropout": 0.2, "fine_tune_at": 150},
        {"lr": 1e-3, "dropout": 0.4, "fine_tune_at": 100},
        {"lr": 5e-4, "dropout": 0.3, "fine_tune_at": 120},
        {"lr": 5e-4, "dropout": 0.5, "fine_tune_at": 80},
        {"lr": 1e-4, "dropout": 0.2, "fine_tune_at": 100},
        {"lr": 1e-4, "dropout": 0.4, "fine_tune_at": 60},
        {"lr": 5e-5, "dropout": 0.3, "fine_tune_at": 80},
        {"lr": 5e-5, "dropout": 0.5, "fine_tune_at": 50},
    ]

    results = []
    print("\n" + "="*65)
    print("    TASK 2.7: VISIONGUARD HYPERPARAMETER TUNING (8 TRIALS)")
    print("="*65)

    for trial_idx, hp in enumerate(search_space, 1):
        model, _ = build_transfer_model(fine_tune_at=hp["fine_tune_at"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp["lr"]),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)
        val_acc = history.history["val_accuracy"][-1]
        val_loss = history.history["val_loss"][-1]

        res = {
            "Trial": trial_idx,
            "LR": hp["lr"],
            "Dropout": hp["dropout"],
            "FineTuneAt": hp["fine_tune_at"],
            "Val_Accuracy": round(val_acc, 4),
            "Val_Loss": round(val_loss, 4)
        }
        results.append(res)
        print(f"Trial {trial_idx}/8: LR={hp['lr']}, Dropout={hp['dropout']}, FineTuneAt={hp['fine_tune_at']} | Val_Acc={val_acc:.4f}, Val_Loss={val_loss:.4f}")

    df_results = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    df_results.to_csv("reports/tuning_results.csv", index=False)
    print("\n--- Tuning Results Summary ---")
    print(df_results.to_string(index=False))
    print("="*65 + "\n")

if __name__ == "__main__":
    run_toy_fashion_mnist()
    run_visionguard_tuning()
