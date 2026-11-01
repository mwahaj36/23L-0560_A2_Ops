import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(audit_csv="reports/dataset_audit.csv", output_dir="data/splits", seed=42):
    df = pd.read_csv(audit_csv)
    df = df[df["format"].isin(["JPEG", "PNG", "WEBP", "MPO", "BMP", "JPG"])].copy()
    
    label_map = {"real": 0, "ai": 1}
    df["label"] = df["class"].map(label_map)
    
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"]
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"]
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[OK] Splits saved to {output_dir}")
    print(f"Train samples: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
    print(f"Val samples:   {len(val_df)} ({val_df['label'].value_counts().to_dict()})")
    print(f"Test samples:  {len(test_df)} ({test_df['label'].value_counts().to_dict()})")

if __name__ == "__main__":
    create_splits()

