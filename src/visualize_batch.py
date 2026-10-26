"""Plot one genuinely augmented training batch."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import create_splits, make_dataset

ROOT = Path(__file__).resolve().parents[1]
splits = create_splits(ROOT / "reports/figures/audit.csv", ROOT / "reports/splits")
batch = make_dataset(splits["train"], ROOT / "data_extracted", training=True, preprocessing="unit")
images, labels = next(iter(batch))
fig, axes = plt.subplots(4, 4, figsize=(8, 8), constrained_layout=True)
for axis, image, label in zip(axes.flat, images[:16], labels[:16]):
    axis.imshow(np.clip(image.numpy(), 0, 1))
    axis.set_title("AI" if int(label) else "Real")
    axis.axis("off")
fig.suptitle("Training batch after flip, rotation, and zoom")
fig.savefig(ROOT / "reports/figures/augmented_batch.png", dpi=150)
print({name: len(path.read_text().splitlines()) - 1 for name, path in splits.items()})
