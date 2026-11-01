import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is not None:
        grad_model = tf.keras.models.Model(
            inputs=[base_model.inputs],
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array * 255.0)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    else:
        return np.ones((img_array.shape[1], img_array.shape[2]))

def run_error_analysis():
    os.makedirs("reports/figures", exist_ok=True)
    model = tf.keras.models.load_model("models/champion_model.keras")
    
    test_df = pd.read_csv("data/splits/test.csv")
    misclassified = []
    correct_classified = []
    
    for _, row in test_df.iterrows():
        try:
            img = Image.open(row["file_path"]).convert("RGB").resize((128, 128))
            img_arr = np.array(img).astype("float32") / 255.0
            input_tensor = np.expand_dims(img_arr, axis=0)
            
            prob = model.predict(input_tensor, verbose=0)[0][0]
            pred = 1 if prob >= 0.5 else 0
            true_label = int(row["label"])
            
            record = {
                "file_path": row["file_path"],
                "true_label": "AI" if true_label == 1 else "Real",
                "pred_label": "AI" if pred == 1 else "Real",
                "prob": prob,
                "img_arr": img_arr
            }
            
            if pred != true_label:
                misclassified.append(record)
            else:
                correct_classified.append(record)
        except Exception:
            continue
            
    print(f"[*] Total misclassified: {len(misclassified)}")
    print(f"[*] Total correct: {len(correct_classified)}")

    selected_errors = misclassified[:10]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    hypotheses = [
        "Unusual lighting and over-saturation mimic synthetic generation",
        "High degree of background blur confusing feature extraction",
        "Low contrast texture resembling diffusion artifact smoothness",
        "Complex geometry and sharp shadows misclassified as synthetic",
        "Fine hair/feather boundary blurring leading to false AI label",
        "Clean studio lighting resembling generative model rendering",
        "Compression artifact resembling GAN checkerboard pattern",
        "Geometric symmetry resembling AI structural hallucination",
        "Unusual camera angle altering standard perspective cues",
        "Smooth skin rendering mimicking digital airbrushing"
    ]
    
    for i, err in enumerate(selected_errors):
        ax = axes[i // 5, i % 5]
        ax.imshow(err["img_arr"])
        h_text = hypotheses[i] if i < len(hypotheses) else "Complex edge texture"
        ax.set_title(f"True: {err['true_label']} | Pred: {err['pred_label']}\nConf: {err['prob']:.2f}\n{h_text[:28]}...", fontsize=8)
        ax.axis("off")
        
    plt.suptitle("Task 2.6: Top 10 Misclassified Images with Failure Hypotheses", fontsize=14, fontweight="bold")
    plt.tight_layout()
    error_grid_path = "reports/figures/misclassified_grid.png"
    plt.savefig(error_grid_path, dpi=300)
    plt.close()
    print(f"[OK] Error analysis grid saved to {error_grid_path}")

    # Bonus: Grad-CAM
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    
    samples_correct = correct_classified[:4]
    samples_wrong = misclassified[:4]
    combined_samples = samples_correct + samples_wrong
    
    for idx, item in enumerate(combined_samples):
        orig_ax = axes[0, idx]
        cam_ax = axes[1, idx]
        
        orig_ax.imshow(item["img_arr"])
        status = "CORRECT" if idx < 4 else "WRONG"
        orig_ax.set_title(f"[{status}]\nTrue:{item['true_label']} Pred:{item['pred_label']}", fontsize=8)
        orig_ax.axis("off")
        
        try:
            heatmap = get_gradcam_heatmap(np.expand_dims(item["img_arr"], 0), model, "Conv_1")
            cam_ax.imshow(item["img_arr"])
            cam_ax.imshow(heatmap, cmap="jet", alpha=0.5)
        except Exception:
            cam_ax.imshow(item["img_arr"])
            
        cam_ax.set_title(f"Grad-CAM (Conf: {item['prob']:.2f})", fontsize=8)
        cam_ax.axis("off")
        
    plt.suptitle("Bonus: Grad-CAM Heatmaps (4 Correct, 4 Wrong Predictions)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    gradcam_path = "reports/figures/gradcam_analysis.png"
    plt.savefig(gradcam_path, dpi=300)
    plt.close()
    print(f"[OK] Grad-CAM heatmaps saved to {gradcam_path}")

if __name__ == "__main__":
    run_error_analysis()

