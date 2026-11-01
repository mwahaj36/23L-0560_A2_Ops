import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def evaluate_model(model, dataset, output_prefix="baseline"):
    os.makedirs("reports/figures", exist_ok=True)
    
    y_true = []
    y_pred_probs = []
    
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds.flatten())
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_probs)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc
    }
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real (0)", "AI (1)"],
                yticklabels=["Real (0)", "AI (1)"])
    plt.title(f"Confusion Matrix - {output_prefix.capitalize()}", fontsize=12, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = f"reports/figures/{output_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {output_prefix.capitalize()}", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = f"reports/figures/{output_prefix}_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    return metrics, cm_path, roc_path, y_true, y_pred_probs

