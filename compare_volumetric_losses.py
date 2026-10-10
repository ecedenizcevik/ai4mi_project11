#!/usr/bin/env python3
"""Compare saved original-space volumetric segmentation metrics."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "volumetric_comparison"

EXPERIMENTS = {
    "CE": "ece_ce_h100_20epochs",
    "CE + Dice": "ece_ce_dice_20epochs",
    "Tversky": "ece_tversky_20epochs",
    "Focal": "ece_focal_20epochs",
    "CE + Tversky": "ece_ce_tversky_20epochs",
}

ORGANS = ["Esophagus", "Heart", "Trachea", "Aorta"]
METRICS = ["dice", "prec", "rec", "hd95", "assd", "nsd", "cldice"]


def load_experiment(name, directory):
    folder = RESULTS_DIR / directory / "volumes"
    missing = [
        metric for metric in METRICS
        if not (folder / f"{metric}.npz").exists()
    ]

    if missing:
        print(f"Skipping {name}: missing {', '.join(missing)}")
        return None

    results = {}

    for metric in METRICS:
        with np.load(folder / f"{metric}.npz") as data:
            results[metric] = {
                patient: np.asarray(data[patient], dtype=float)
                for patient in data.files
            }

    expected_patients = set(results["dice"])

    for metric, patient_results in results.items():
        if set(patient_results) != expected_patients:
            raise ValueError(
                f"{name}: inconsistent patients in {metric}"
            )

        for patient, values in patient_results.items():
            if values.shape != (len(ORGANS),):
                raise ValueError(
                    f"{name}, {patient}, {metric}: "
                    f"unexpected shape {values.shape}"
                )

    print(f"Loaded {name}: {len(expected_patients)} patients")
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, directory in EXPERIMENTS.items():
        result = load_experiment(name, directory)
        if result is not None:
            all_results[name] = result

    if not all_results:
        raise RuntimeError("No complete evaluation results found")

    # Verify that all included losses use the same patients.
    reference = set(
        next(iter(all_results.values()))["dice"]
    )

    for name, results in all_results.items():
        if set(results["dice"]) != reference:
            raise ValueError(
                f"{name}: patient set differs from other experiments"
            )

    patient_rows = []
    summary_rows = []

    for loss, results in all_results.items():
        patients = sorted(results["dice"])

        for patient in patients:
            for organ_idx, organ in enumerate(ORGANS):
                row = {
                    "loss": loss,
                    "patient": patient,
                    "organ": organ,
                }

                for metric in METRICS:
                    row[metric] = float(
                        results[metric][patient][organ_idx]
                    )

                patient_rows.append(row)

        for organ_idx, organ in enumerate(ORGANS):
            for metric in METRICS:
                values = np.array([
                    results[metric][patient][organ_idx]
                    for patient in patients
                ])

                summary_rows.append({
                    "loss": loss,
                    "organ": organ,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values)),
                    "n_patients": len(patients),
                })

    patient_file = OUTPUT_DIR / "volumetric_loss_per_patient.csv"

    with patient_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["loss", "patient", "organ"] + METRICS,
        )
        writer.writeheader()
        writer.writerows(patient_rows)

    summary_file = OUTPUT_DIR / "volumetric_loss_summary.csv"

    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "loss", "organ", "metric",
                "mean", "sd", "n_patients",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Grouped bar chart of organ Dice.
    losses = list(all_results)
    x = np.arange(len(ORGANS))
    width = 0.8 / len(losses)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, loss in enumerate(losses):
        values = [
            next(
                row["mean"] for row in summary_rows
                if row["loss"] == loss
                and row["organ"] == organ
                and row["metric"] == "dice"
            )
            for organ in ORGANS
        ]

        offset = (i - (len(losses) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=loss)

    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS)
    ax.set_ylabel("Mean 3D Dice")
    ax.set_ylim(0, 1)
    ax.set_title("Volumetric segmentation Dice by loss function")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    plot_file = OUTPUT_DIR / "volumetric_dice_comparison.png"
    fig.savefig(plot_file, dpi=200)
    plt.close(fig)

    print("\nMean foreground Dice:")
    for loss, results in all_results.items():
        values = np.array([
            results["dice"][patient]
            for patient in sorted(results["dice"])
        ])
        print(f"  {loss:15s}: {values.mean():.4f}")

    print("\nSaved:")
    print(f"  {patient_file}")
    print(f"  {summary_file}")
    print(f"  {plot_file}")


if __name__ == "__main__":
    main()
