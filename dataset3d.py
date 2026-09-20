#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026

"""Volume-level dataset for 3D segmentation.

Reads NIfTI directly instead of the pre-sliced PNGs, because a 3D network needs
the depth axis intact. Resampling to a uniform spacing follows Isensee et al.
(nnU-Net, Nature Methods 2021): convolutions operate on voxel grids and ignore
physical spacing, so heterogeneous spacings mean the same kernel covers
different physical distances in different patients.
"""

import random
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from torch import Tensor
from torch.utils.data import Dataset

from slice_segthor import norm_arr
from utils import class2one_hot

# Median spacing over the training cases (mm). Anisotropy is 2.6 at most, below
# nnU-Net's threshold of 3, so the plain per-axis median applies.
TARGET_SPACING: tuple[float, float, float] = (0.977, 0.977, 2.5)


def resample(vol: np.ndarray, spacing, target, is_label: bool, K: int = 5) -> np.ndarray:
    factors = tuple(s / t for s, t in zip(spacing, target))
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return vol

    if not is_label:
        # order 3 in-plane, order 1 through-plane: contours change a lot between
        # slices, so high-order interpolation there invites artefacts.
        return zoom(vol, factors, order=3, mode='nearest')

    # Labels: interpolate each class separately, then argmax back to a mask.
    # Interpolating label indices directly would invent values between classes.
    stack = np.stack([zoom((vol == k).astype(np.float32), factors, order=1, mode='nearest')
                      for k in range(K)], axis=0)
    return stack.argmax(axis=0).astype(np.uint8)


class VolumeDataset(Dataset):
    def __init__(self, ids: list[str], root_dir: Path, patch_size=(128, 128, 32),
                 patches_per_epoch: int = 250, fg_ratio: float = 1 / 3,
                 K: int = 5, train: bool = True, seed: int = 0):
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.fg_ratio = fg_ratio
        self.K = K
        self.train = train
        self.rng = random.Random(seed)

        self.volumes: list[tuple[np.ndarray, np.ndarray]] = []
        self.fg_voxels: list[np.ndarray] = []

        for pid in ids:
            d = Path(root_dir) / 'train' / pid
            img_nii = nib.load(str(d / f"{pid}.nii.gz"))
            gt_nii = nib.load(str(d / "GT.nii.gz"))
            spacing = img_nii.header.get_zooms()[:3]

            img = resample(img_nii.get_fdata(), spacing, TARGET_SPACING, is_label=False)
            gt = resample(gt_nii.get_fdata().astype(np.uint8), spacing, TARGET_SPACING,
                          is_label=True, K=K)

            # Same intensity handling as the 2D baseline: whole-volume min/max to
            # uint8, then to [0, 1]. Deliberately unchanged, so this experiment
            # varies the architecture only.
            img = norm_arr(img).astype(np.float32) / 255

            assert img.shape == gt.shape, (img.shape, gt.shape)
            self.volumes.append((img, gt))
            self.fg_voxels.append(np.argwhere(gt > 0))
            print(f">> Loaded {pid}: {img.shape}, {len(self.fg_voxels[-1])} foreground voxels")

        print(f">> Created {'train' if train else 'val'} dataset: {len(self.volumes)} volumes")

    def __len__(self) -> int:
        return self.patches_per_epoch

    def _sample_origin(self, vol_idx: int) -> tuple[int, int, int]:
        shape = self.volumes[vol_idx][0].shape
        pw, ph, pd = self.patch_size
        limits = [max(0, s - p) for s, p in zip(shape, (pw, ph, pd))]

        if self.rng.random() < self.fg_ratio and len(self.fg_voxels[vol_idx]) > 0:
            # Centre the patch on a random foreground voxel, then clamp so the
            # patch stays inside the volume.
            centre = self.fg_voxels[vol_idx][self.rng.randrange(len(self.fg_voxels[vol_idx]))]
            return tuple(int(np.clip(c - p // 2, 0, lim))
                         for c, p, lim in zip(centre, (pw, ph, pd), limits))

        return tuple(self.rng.randint(0, lim) for lim in limits)

    def __getitem__(self, index) -> dict[str, Tensor | str]:
        vol_idx = self.rng.randrange(len(self.volumes))
        img, gt = self.volumes[vol_idx]
        x, y, z = self._sample_origin(vol_idx)
        pw, ph, pd = self.patch_size

        img_patch = img[x:x + pw, y:y + ph, z:z + pd]
        gt_patch = gt[x:x + pw, y:y + ph, z:z + pd]

        # Volumes near the edge of the grid can come up short; pad to a fixed size
        # so the batch collates.
        pad = [(0, p - s) for s, p in zip(img_patch.shape, (pw, ph, pd))]
        if any(b > 0 for _, b in pad):
            img_patch = np.pad(img_patch, pad, mode='constant')
            gt_patch = np.pad(gt_patch, pad, mode='constant')

        # [W, H, D] -> [C, D, H, W] to match PyTorch's Conv3d layout
        image = torch.from_numpy(np.ascontiguousarray(img_patch.transpose(2, 1, 0)))[None].float()
        label = torch.from_numpy(np.ascontiguousarray(gt_patch.transpose(2, 1, 0)))[None].long()

        return {"images": image,
                "gts": class2one_hot(label, K=self.K)[0],
                "stems": f"vol{vol_idx:02d}_{x}_{y}_{z}"}