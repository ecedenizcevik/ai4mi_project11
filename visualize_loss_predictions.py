
#!/usr/bin/env python3
"""Visualize original-space CT segmentation predictions across losses."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import nibabel as nib
import numpy as np


EXPERIMENTS = {
    "CE + Tversky": "ece_ce_tversky_20epochs",
    "CE + Dice": "ece_ce_dice_20epochs",
    "Focal": "ece_focal_20epochs",
    "Tversky": "ece_tversky_20epochs",
}

# Labels: 0 background, 1 esophagus, 2 heart,
# 3 trachea, 4 aorta.
COLORS = ["none", "#ffcc00", "#f04452", "#00c8ff", "#53d769"]
ORGAN_NAMES = ["Esophagus", "Heart", "Trachea", "Aorta"]


def load_volume(path):
    return np.asarray(nib.load(str(path)).dataobj)


def choose_slice(gt):
    """Select a slice containing both esophagus and heart."""
    candidates = []

    for z in range(gt.shape[2]):
        esophagus = np.count_nonzero(gt[:, :, z] == 1)
        heart = np.count_nonzero(gt[:, :, z] == 2)

        if esophagus > 0 and heart > 0:
            candidates.append((heart + 3 * esophagus, z))

    if candidates:
        return max(candidates)[1]

    return int(np.argmax(np.sum(gt > 0, axis=(0, 1))))


def overlay(ax, ct_slice, label_slice, vmin, vmax):
    ax.imshow(
        ct_slice.T,
        cmap="gray",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    # Transparent background and colored organ labels.
    color_map = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), 5)

    masked = np.ma.masked_where(label_slice == 0, label_slice)

    ax.imshow(
        masked.T,
        cmap=color_map,
        norm=norm,
        origin="lower",
        alpha=0.48,
        interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="Patient_28")
    parser.add_argument("--slice", type=int, default=None)
    args = parser.parse_args()

    patient = args.patient
    root = Path("data/segthor_train_full/train") / patient

    ct = load_volume(root / f"{patient}.nii.gz")
    gt = load_volume(root / "GT.nii.gz")

    predictions = {}

    for loss, directory in EXPERIMENTS.items():
        path = (
            Path("results")
            / directory
            / "volumes"
            / f"{patient}.nii.gz"
        )
        predictions[loss] = load_volume(path)

    for name, volume in [("GT", gt), *predictions.items()]:
        if volume.shape != ct.shape:
            raise ValueError(
                f"{name}: shape {volume.shape} differs from CT {ct.shape}"
            )

    z = choose_slice(gt) if args.slice is None else args.slice

    if not 0 <= z < ct.shape[2]:
        raise ValueError(f"Invalid slice index: {z}")

    # Fixed CT window for consistent display across models.
    ct_slice = np.clip(ct[:, :, z], -160, 240)

    panels = [
        ("Original CT", None),
        ("Ground Truth", gt),
        *predictions.items(),
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 10),
        constrained_layout=True,
    )

    for ax, (title, labels) in zip(axes.flat, panels):
        if labels is None:
            overlay(
                ax, ct_slice,
                np.zeros_like(ct_slice),
                -160, 240
            )
        else:
            overlay(
                ax, ct_slice,
                labels[:, :, z],
                -160, 240
            )

        ax.set_title(title, fontsize=14)

    from matplotlib.patches import Patch

    legend = [
        Patch(color=COLORS[i], label=ORGAN_NAMES[i - 1])
        for i in range(1, 5)
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"{patient} — Slice {z}: Segmentation comparison",
        fontsize=16,
    )

    output_dir = Path("results/volumetric_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"{patient}_slice_{z}_comparison.png"

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Selected slice: {z}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
