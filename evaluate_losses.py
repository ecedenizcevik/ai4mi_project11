
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = Path("results")

EXPERIMENTS = {
    "Tversky": ROOT / "ece_tversky_20epochs",
    "Focal": ROOT / "ece_focal_20epochs",
}

GT_DIR = Path("data/SEGTHOR_FULL/val/gt")

ORGANS = {
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}


def calculate_metrics(tp, fp, fn):
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan

    return dice, precision, recall


def evaluate(name, experiment_dir):
    pred_dir = experiment_dir / "best_epoch" / "val"

    pred_files = sorted(pred_dir.glob("*.png"))
    gt_files = sorted(GT_DIR.glob("*.png"))

    pred_names = {f.name for f in pred_files}
    gt_names = {f.name for f in gt_files}

    if pred_names != gt_names:
        raise RuntimeError(
            f"{name}: prediction/GT mismatch. "
            f"Missing predictions: {len(gt_names - pred_names)}; "
            f"unexpected predictions: {len(pred_names - gt_names)}"
        )

    if not pred_files:
        raise RuntimeError(f"{name}: no prediction files found")

    patients = defaultdict(lambda: np.zeros((5, 5), dtype=np.int64))

    for pred_file in pred_files:
        gt_file = GT_DIR / pred_file.name

        pred_raw = np.asarray(Image.open(pred_file))
        gt_raw = np.asarray(Image.open(gt_file))

        if pred_raw.shape != gt_raw.shape:
            raise ValueError(f"Shape mismatch: {pred_file.name}")

        if not np.isin(pred_raw, [0, 63, 126, 189, 252]).all():
            raise ValueError(f"Unexpected prediction labels: {pred_file}")

        if not np.isin(gt_raw, [0, 63, 126, 189, 252]).all():
            raise ValueError(f"Unexpected GT labels: {gt_file}")

        pred = pred_raw.astype(np.int64) // 63
        gt = gt_raw.astype(np.int64) // 63

        patient = "_".join(pred_file.stem.split("_")[:2])

        counts = np.bincount(
            gt.ravel() * 5 + pred.ravel(),
            minlength=25
        ).reshape(5, 5)

        patients[patient] += counts

    print(f"\n{'=' * 65}")
    print(f"{name.upper()} — BEST CHECKPOINT")
    print(f"Validation slices: {len(pred_files)}")
    print(f"Validation patients: {len(patients)}")
    print(f"{'=' * 65}")

    organ_results = {}

    for k, organ in ORGANS.items():
        scores = []

        for patient, matrix in sorted(patients.items()):
            tp = matrix[k, k]
            fp = matrix[:, k].sum() - tp
            fn = matrix[k, :].sum() - tp

            dice, precision, recall = calculate_metrics(tp, fp, fn)

            scores.append([dice, precision, recall])

        scores = np.asarray(scores, dtype=float)

        mean_scores = np.array([
            np.nanmean(scores[:, i])
            if not np.isnan(scores[:, i]).all()
            else np.nan
            for i in range(3)
        ])

        organ_results[organ] = mean_scores

        print(
            f"{organ:10s} | "
            f"Dice: {mean_scores[0]:.4f} | "
            f"Precision: {mean_scores[1]:.4f} | "
            f"Recall: {mean_scores[2]:.4f}"
        )

    mean_dice = np.nanmean([
        values[0] for values in organ_results.values()
    ])

    print(f"\nMean foreground Dice: {mean_dice:.4f}")

    return organ_results


if __name__ == "__main__":
    all_results = {}

    for name, path in EXPERIMENTS.items():
        all_results[name] = evaluate(name, path)

    print("\n" + "=" * 65)
    print("COMPARISON: FOCAL MINUS TVERSKY")
    print("=" * 65)

    for organ in ORGANS.values():
        tversky = all_results["Tversky"][organ][0]
        focal = all_results["Focal"][organ][0]

        print(
            f"{organ:10s} | "
            f"Tversky: {tversky:.4f} | "
            f"Focal: {focal:.4f} | "
            f"Difference: {focal - tversky:+.4f}"
        )
