#!/usr/bin/env python3

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, generate_binary_structure


CLASSES = {
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}

# SEGTHOR PNG encoding used by this project:
# 0, 63, 126, 189, 252 -> classes 0, 1, 2, 3, 4
SCALE = 63


def read_seg(path):
    arr = np.asarray(Image.open(path), dtype=np.float32)
    seg = np.rint(arr / SCALE).astype(np.uint8)

    values = set(np.unique(seg).tolist())
    valid = set(range(5))

    if not values.issubset(valid):
        raise ValueError(
            f"Unexpected labels in {path}: {sorted(values)}"
        )

    return seg


def parse_stem(path):
    match = re.match(r"^(Patient_\d+)_(\d+)$", path.stem)

    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")

    patient = match.group(1)
    slice_idx = int(match.group(2))

    return patient, slice_idx


def load_volumes(pred_dir, gt_dir):
    grouped = defaultdict(list)

    for pred_path in pred_dir.glob("*.png"):
        patient, slice_idx = parse_stem(pred_path)

        gt_path = gt_dir / pred_path.name

        if not gt_path.exists():
            raise FileNotFoundError(
                f"Missing GT for {pred_path.name}"
            )

        grouped[patient].append(
            (slice_idx, pred_path, gt_path)
        )

    volumes = {}

    for patient, entries in grouped.items():
        entries = sorted(entries, key=lambda x: x[0])

        pred_slices = []
        gt_slices = []

        for _, pred_path, gt_path in entries:
            pred_slices.append(read_seg(pred_path))
            gt_slices.append(read_seg(gt_path))

        pred = np.stack(pred_slices, axis=0)
        gt = np.stack(gt_slices, axis=0)

        assert pred.shape == gt.shape

        volumes[patient] = (pred, gt)

    return volumes


def remove_small_components(pred, min_size):
    if min_size == 0:
        return pred.copy()

    output = pred.copy()

    # 26-connected neighbourhood in 3D.
    structure = generate_binary_structure(
        rank=3,
        connectivity=3
    )

    for cls in CLASSES:
        mask = pred == cls

        components, n_components = label(
            mask,
            structure=structure
        )

        if n_components == 0:
            continue

        sizes = np.bincount(components.ravel())

        # Component 0 is background.
        small_ids = np.where(sizes < min_size)[0]
        small_ids = small_ids[small_ids != 0]

        if len(small_ids) == 0:
            continue

        remove_mask = np.isin(
            components,
            small_ids
        )

        output[remove_mask] = 0

    return output


def dice_binary(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    denominator = pred.sum() + gt.sum()

    # Match the project's Dice behaviour for empty-empty masks.
    if denominator == 0:
        return 1.0

    intersection = np.logical_and(pred, gt).sum()

    return 2.0 * intersection / denominator


def evaluate(volumes, min_size):
    slice_scores = defaultdict(list)
    volume_scores = defaultdict(list)

    removed_voxels = 0

    for patient, (raw_pred, gt) in volumes.items():
        pred = remove_small_components(
            raw_pred,
            min_size=min_size
        )

        removed_voxels += int(
            np.sum(raw_pred != pred)
        )

        for cls in CLASSES:
            # Patient-level 3D Dice
            volume_scores[cls].append(
                dice_binary(
                    pred == cls,
                    gt == cls
                )
            )

            # Slice-wise Dice, matching the existing
            # training/validation evaluation style.
            for z in range(pred.shape[0]):
                slice_scores[cls].append(
                    dice_binary(
                        pred[z] == cls,
                        gt[z] == cls
                    )
                )

    slice_per_class = {
        cls: float(np.mean(slice_scores[cls]))
        for cls in CLASSES
    }

    volume_per_class = {
        cls: float(np.mean(volume_scores[cls]))
        for cls in CLASSES
    }

    return {
        "min_size": min_size,
        "slice_mean": float(
            np.mean(list(slice_per_class.values()))
        ),
        "volume_mean": float(
            np.mean(list(volume_per_class.values()))
        ),
        "slice_per_class": slice_per_class,
        "volume_per_class": volume_per_class,
        "removed_voxels": removed_voxels,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--gt_dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[0, 25, 50, 100, 250, 500]
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("postprocess_cc_results.csv")
    )

    args = parser.parse_args()

    volumes = load_volumes(
        args.pred_dir,
        args.gt_dir
    )

    print(
        f"Loaded {len(volumes)} patients: "
        f"{sorted(volumes.keys())}"
    )

    results = []

    print()
    print(
        f"{'Min size':>8} "
        f"{'Slice DSC':>10} "
        f"{'3D DSC':>10} "
        f"{'Eso':>8} "
        f"{'Heart':>8} "
        f"{'Trachea':>8} "
        f"{'Aorta':>8} "
        f"{'Removed':>10}"
    )

    for threshold in args.thresholds:
        result = evaluate(
            volumes,
            min_size=threshold
        )

        results.append(result)

        pc = result["volume_per_class"]

        print(
            f"{threshold:8d} "
            f"{result['slice_mean']:10.4f} "
            f"{result['volume_mean']:10.4f} "
            f"{pc[1]:8.4f} "
            f"{pc[2]:8.4f} "
            f"{pc[3]:8.4f} "
            f"{pc[4]:8.4f} "
            f"{result['removed_voxels']:10d}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "min_size",
            "slice_mean_dsc",
            "volume_mean_dsc",
            "esophagus_3d_dsc",
            "heart_3d_dsc",
            "trachea_3d_dsc",
            "aorta_3d_dsc",
            "removed_voxels",
        ])

        for result in results:
            pc = result["volume_per_class"]

            writer.writerow([
                result["min_size"],
                result["slice_mean"],
                result["volume_mean"],
                pc[1],
                pc[2],
                pc[3],
                pc[4],
                result["removed_voxels"],
            ])

    print()
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
