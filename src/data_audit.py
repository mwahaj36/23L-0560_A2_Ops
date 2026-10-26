"""Audit source images and produce reproducible dataset evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

CLASSES = {"AiArtData": "AI", "RealArt": "Real"}


def audit(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    corrupt = []
    for folder, label in CLASSES.items():
        for path in sorted((root / folder / folder).rglob("*")):
            if not path.is_file():
                continue
            try:
                payload = path.read_bytes()
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    mode, fmt = image.mode, image.format
                rows.append({"path": str(path.relative_to(root)).replace("\\", "/"),
                             "label": label, "bytes": len(payload), "width": width,
                             "height": height, "mode": mode, "format": fmt,
                             "sha256": hashlib.sha256(payload).hexdigest()})
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                corrupt.append({"path": str(path.relative_to(root)), "error": str(exc)})
    with (output / "audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hashes = defaultdict(list)
    for row in rows:
        hashes[row["sha256"]].append(row["path"])
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    summary = {"total": len(rows), "corrupt": corrupt, "duplicate_groups": duplicates, "classes": {}}
    for label in CLASSES.values():
        subset = [row for row in rows if row["label"] == label]
        summary["classes"][label] = {
            "count": len(subset),
            "width": stats(subset, "width"), "height": stats(subset, "height"),
            "bytes": stats(subset, "bytes"),
            "modes": dict(Counter(row["mode"] for row in subset)),
            "formats": dict(Counter(row["format"] for row in subset)),
        }
    (output / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def stats(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def plot_eda(root: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(output / "audit.csv")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for idx, (folder, label) in enumerate(CLASSES.items()):
        subset = frame[frame.label == label]
        chosen = subset.sample(n=min(6, len(subset)), random_state=560)
        tiles = []
        for item in chosen.itertuples():
            with Image.open(root / item.path) as image:
                tile = image.convert("RGB")
                tile.thumbnail((160, 120))
                from PIL import ImageOps
                tiles.append(ImageOps.pad(tile, (160, 120)))
        grid = Image.new("RGB", (160 * 3, 120 * 2), "white")
        for n, tile in enumerate(tiles):
            grid.paste(tile, ((n % 3) * 160, (n // 3) * 120))
        axes[idx, 0].imshow(grid)
        axes[idx, 0].set_title(f"{label}: sample images")
        axes[idx, 0].axis("off")
        for j, (column, unit) in enumerate((("width", "px"), ("height", "px"), ("bytes", "bytes")), 1):
            axes[idx, j].hist(subset[column], bins=25, color="#314f9f" if idx == 0 else "#d67736")
            axes[idx, j].set_title(f"{label}: {column} ({unit})")
            axes[idx, j].set_ylabel("images")
    fig.savefig(output / "eda_plots.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_extracted"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    result = audit(args.root, args.output)
    plot_eda(args.root, args.output)
    print(json.dumps(result, indent=2))
