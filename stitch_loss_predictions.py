
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import nibabel as nib
from PIL import Image
from skimage.transform import resize


VALID_VALUES = np.array([0, 63, 126, 189, 252])


def stitch(pred_dir, source_pattern, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    patients = defaultdict(list)

    for path in sorted(pred_dir.glob("*.png")):
        parts = path.stem.split("_")
        patient = "_".join(parts[:2])
        z = int(parts[-1])

        patients[patient].append((z, path))

    if not patients:
        raise RuntimeError("No PNG predictions found")

    for patient, slices in sorted(patients.items()):
        original = nib.load(
            source_pattern.format(id_=patient)
        )

        shape = original.shape
        X, Y, Z = shape

        if len(slices) != Z:
            raise RuntimeError(
                f"{patient}: expected {Z} slices, found {len(slices)}"
            )

        volume = np.zeros(shape, dtype=np.uint8)
        seen = set()

        for z, png_path in slices:
            if z in seen or not 0 <= z < Z:
                raise ValueError(
                    f"{patient}: invalid/duplicate slice index {z}"
                )
            seen.add(z)

            raw = np.asarray(Image.open(png_path))

            if not np.isin(raw, VALID_VALUES).all():
                raise ValueError(
                    f"Unexpected labels in {png_path}"
                )

            labels = (raw.astype(np.uint16) // 63).astype(
                np.uint8
            )

            restored = resize(
                labels,
                (X, Y),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.uint8)

            volume[:, :, z] = restored

        if seen != set(range(Z)):
            raise RuntimeError(
                f"{patient}: missing slice indices"
            )

        header = original.header.copy()
        header.set_data_dtype(np.uint8)

        output = nib.Nifti1Image(
            volume,
            affine=original.affine,
            header=header
        )

        destination = output_dir / f"{patient}.nii.gz"
        nib.save(output, destination)

        print(
            f"{patient}: {volume.shape}, "
            f"classes={np.unique(volume).tolist()}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--source_pattern", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    args = parser.parse_args()

    stitch(
        args.pred_dir,
        args.source_pattern,
        args.output_dir
    )
