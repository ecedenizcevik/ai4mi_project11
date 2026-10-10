
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

GT_DIR = Path("data/SEGTHOR_FULL/val/gt")

EXPERIMENTS = {
    "CE Baseline": Path("results/ece_ce_20epochs"),
    "CE + Dice": Path("results/ece_ce_dice_20epochs"),
    "Tversky": Path("results/ece_tversky_20epochs"),
    "Focal": Path("results/ece_focal_20epochs"),
    "CE + Tversky": Path("results/ece_ce_tversky_20epochs"),
}

ORGANS = ["Esophagus", "Heart", "Trachea", "Aorta"]
VALID_LABELS = np.array([0, 63, 126, 189, 252])


def evaluate_epoch(pred_dir):
    pred_files = sorted(pred_dir.glob("*.png"))
    gt_files = sorted(GT_DIR.glob("*.png"))

    if {p.name for p in pred_files} != {p.name for p in gt_files}:
        raise RuntimeError(
            f"Missing or unexpected predictions in {pred_dir}"
        )

    patients = defaultdict(
        lambda: np.zeros((5, 5), dtype=np.int64)
    )

    for pred_file in pred_files:
        pred_raw = np.asarray(Image.open(pred_file))
        gt_raw = np.asarray(
            Image.open(GT_DIR / pred_file.name)
        )

        if pred_raw.shape != gt_raw.shape:
            raise ValueError(f"Shape mismatch: {pred_file.name}")

        if not np.isin(pred_raw, VALID_LABELS).all():
            raise ValueError(f"Unexpected prediction labels: {pred_file}")

        if not np.isin(gt_raw, VALID_LABELS).all():
            raise ValueError(f"Unexpected GT labels: {pred_file.name}")

        pred = pred_raw.astype(np.int64) // 63
        gt = gt_raw.astype(np.int64) // 63

        patient = "_".join(pred_file.stem.split("_")[:2])

        patients[patient] += np.bincount(
            gt.ravel() * 5 + pred.ravel(),
            minlength=25
        ).reshape(5, 5)

    if len(patients) != 5:
        raise RuntimeError(
            f"Expected 5 patients, found {len(patients)}"
        )

    organ_dice = []

    for k in range(1, 5):
        patient_dice = []

        for matrix in patients.values():
            tp = matrix[k, k]
            gt_count = matrix[k, :].sum()
            pred_count = matrix[:, k].sum()

            denominator = gt_count + pred_count

            if denominator > 0:
                patient_dice.append(2 * tp / denominator)
            else:
                patient_dice.append(np.nan)

        organ_dice.append(np.nanmean(patient_dice))

    return np.array(organ_dice)


def main():
    for name, experiment_dir in EXPERIMENTS.items():
        print(f"\n{'=' * 75}")
        print(f"{name.upper()} — PATIENT-LEVEL DICE BY EPOCH")
        print(f"{'=' * 75}")

        results = []

        for epoch in range(20):
            pred_dir = (
                experiment_dir / f"iter{epoch:03d}" / "val"
            )

            scores = evaluate_epoch(pred_dir)
            mean_dice = np.nanmean(scores)

            results.append([epoch, *scores, mean_dice])

            print(
                f"Epoch {epoch:02d} | "
                f"Mean: {mean_dice:.4f} | "
                + " | ".join(
                    f"{organ}: {score:.4f}"
                    for organ, score in zip(ORGANS, scores)
                )
            )

        results = np.asarray(results)

        best_index = int(np.nanargmax(results[:, -1]))
        best = results[best_index]

        print(f"\nBEST PATIENT-LEVEL EPOCH: {int(best[0])}")
        print(f"BEST PATIENT-LEVEL MEAN DICE: {best[-1]:.4f}")

        for organ, score in zip(ORGANS, best[1:5]):
            print(f"  {organ}: {score:.4f}")


if __name__ == "__main__":
    main()
