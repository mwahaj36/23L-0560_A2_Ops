import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

def get_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

def parse_image(file_path, label, img_size=(128, 128)):
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def create_dataset_from_csv(csv_path, img_size=(128, 128), batch_size=32, is_training=False, augment=True):
    df = pd.read_csv(csv_path)
    file_paths = df["file_path"].values
    labels = df["label"].values.astype("float32")
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=42)
    
    ds = ds.map(lambda fp, lbl: parse_image(fp, lbl, img_size=img_size), num_parallel_calls=AUTOTUNE)
    
    if is_training and augment:
        augmentation = get_data_augmentation()
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def save_augmented_batch_figure(csv_path="data/splits/train.csv", output_path="reports/figures/augmented_batch.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds = create_dataset_from_csv(csv_path, batch_size=8, is_training=True, augment=True)
    images, labels = next(iter(ds))
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i].numpy())
        label_text = "AI (1)" if int(labels[i].numpy()) == 1 else "Real (0)"
        ax.set_title(label_text, fontsize=10)
        ax.axis("off")
        
    plt.suptitle("Augmented Training Batch (Task 2.1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Augmented batch figure saved to {output_path}")

if __name__ == "__main__":
    save_augmented_batch_figure()

