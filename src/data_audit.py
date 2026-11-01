import os
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def audit_dataset(data_dir="data/raw", output_csv="reports/dataset_audit.csv"):
    data_path = Path(data_dir)
    records = []
    corrupt_files = []
    
    classes = [d.name for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    print(f"[*] Found classes: {classes}")
    
    for cls in classes:
        cls_dir = data_path / cls
        files = list(cls_dir.rglob("*"))
        print(f"[*] Scanning {cls}: {len(files)} files found...")
        
        for file in files:
            if not file.is_file():
                continue
            
            file_size_bytes = file.stat().st_size
            file_size_kb = file_size_bytes / 1024.0
            
            try:
                file_hash = compute_sha256(file)
            except Exception as e:
                corrupt_files.append((str(file), f"Read error: {e}"))
                continue
            
            try:
                with Image.open(file) as img:
                    img.verify()
                with Image.open(file) as img:
                    width, height = img.size
                    mode = img.mode
                    img_format = img.format
                
                records.append({
                    "file_path": str(file),
                    "class": cls,
                    "file_size_bytes": file_size_bytes,
                    "file_size_kb": file_size_kb,
                    "width": width,
                    "height": height,
                    "colour_mode": mode,
                    "format": img_format,
                    "sha256": file_hash
                })
            except Exception as e:
                corrupt_files.append((str(file), f"Corrupt image: {e}"))
                
    df = pd.DataFrame(records)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Full audit saved to: {output_csv}")
    
    duplicate_mask = df.duplicated(subset=["sha256"], keep=False)
    duplicates_df = df[duplicate_mask]
    num_duplicates = len(duplicates_df)
    
    print("\n" + "="*60)
    print("                 DATASET AUDIT SUMMARY TABLE (1.2)")
    print("="*60)
    print(f"Total valid images analyzed: {len(df)}")
    print(f"Total corrupt/unreadable files: {len(corrupt_files)}")
    print(f"Total duplicate images (identical SHA-256): {num_duplicates}")
    
    print("\n--- Image Count per Class ---")
    print(df["class"].value_counts().to_string())
    
    print("\n--- Dimension & File Size Statistics (Min, Median, Max) ---")
    stats = df.groupby("class").agg({
        "width": ["min", "median", "max"],
        "height": ["min", "median", "max"],
        "file_size_kb": ["min", "median", "max"]
    }).round(2)
    print(stats.to_string())
    
    print("\n--- Overall Statistics (Combined) ---")
    overall = pd.DataFrame({
        "Metric": ["Width (px)", "Height (px)", "File Size (KB)"],
        "Min": [df["width"].min(), df["height"].min(), round(df["file_size_kb"].min(), 2)],
        "Median": [df["width"].median(), df["height"].median(), round(df["file_size_kb"].median(), 2)],
        "Max": [df["width"].max(), df["height"].max(), round(df["file_size_kb"].max(), 2)]
    })
    print(overall.to_string(index=False))
    print("="*60 + "\n")
    
    return df, corrupt_files

def generate_visual_eda(df, data_dir="data/raw", figures_dir="reports/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    classes = df["class"].unique()
    num_samples_per_class = 4
    fig, axes = plt.subplots(len(classes), num_samples_per_class, figsize=(14, 7))
    
    for i, cls in enumerate(classes):
        cls_samples = df[df["class"] == cls].sample(num_samples_per_class, random_state=42)
        for j, (_, row) in enumerate(cls_samples.iterrows()):
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            img = Image.open(row["file_path"])
            ax.imshow(img)
            ax.set_title(f"{cls} ({row['width']}x{row['height']})", fontsize=10)
            ax.axis("off")
            
    plt.suptitle("Sample Image Grid per Class (Task 1.3)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    grid_path = os.path.join(figures_dir, "sample_grid.png")
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"[OK] Sample grid saved to: {grid_path}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(data=df, x="width", hue="class", kde=True, ax=axes[0], bins=20, alpha=0.6)
    axes[0].set_title("Width Distribution by Class", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Width (pixels)")
    
    sns.histplot(data=df, x="height", hue="class", kde=True, ax=axes[1], bins=20, alpha=0.6)
    axes[1].set_title("Height Distribution by Class", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Height (pixels)")
    
    sns.histplot(data=df, x="file_size_kb", hue="class", kde=True, ax=axes[2], bins=20, alpha=0.6)
    axes[2].set_title("File Size (KB) Distribution by Class", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("File Size (KB)")
    
    plt.suptitle("Distribution of Width, Height, and File Size per Class (Task 1.3)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    dist_path = os.path.join(figures_dir, "distributions_histograms.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[OK] Histograms saved to: {dist_path}")

if __name__ == "__main__":
    df, corrupt = audit_dataset()
    generate_visual_eda(df)
    print("\n[OK] Audit & EDA Complete!")
