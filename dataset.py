#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import math
from pathlib import Path
import pickle
from typing import Callable, Union

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_ID_REGEX: str = r"^(?P<group>.+)_(?P<idx>\d+)$"
N_COORDS: dict[str, int] = {'none': 0, 'z': 1, 'xy': 2, 'xyz': 3}


def n_input_channels(neighbours: int = 0, coords: str = 'none',
                     fourier_freqs: int = 0) -> int:
    n_coord: int = N_COORDS[coords]
    if fourier_freqs > 0:
        n_coord *= 2 * fourier_freqs
    return (2 * neighbours + 1) + n_coord


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 neighbours: int = 0, coords: str = 'none',
                 fourier_freqs: int = 0, id_regex: str = DEFAULT_ID_REGEX):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.neighbours: int = neighbours
        self.coords: str = coords
        self.fourier_freqs: int = fourier_freqs
        assert neighbours >= 0 and coords in N_COORDS
        self._xy_cache: dict[tuple[int, int], Tensor] = {}

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        spacing_path = Path(root_dir) / 'spacing.pkl'
        if spacing_path.exists():
            with spacing_path.open('rb') as handle:
                self.spacing = pickle.load(handle)
        else:
            self.spacing = {}
        if debug:
            self.files = self.files[:10]

        self._index_volumes(id_regex)

        print(f">> Created {subset} dataset with {len(self)} images, "
              f"{self.n_volumes} volumes, {self.n_channels} input channels")

    def _index_volumes(self, id_regex):
        """Group slices by patient. self.vol[i] = (that patient's slice indices
        in order from top to bottom, where slice i sits among them)."""
        pat = re.compile(id_regex)
        groups: dict = {}
        for i, (path, _) in enumerate(self.files):
            m = pat.match(path.stem)
            key, z = (m.group('group'), int(m.group('idx'))) if m else (path.stem, 0)
            groups.setdefault(key, []).append((z, i))

        self.vol: dict = {}
        for entries in groups.values():
            order = [i for _, i in sorted(entries)]
            for rank, i in enumerate(order):
                self.vol[i] = (order, rank)
        self.n_volumes: int = len(groups)

    @property
    def n_channels(self) -> int:
        return n_input_channels(self.neighbours, self.coords, self.fourier_freqs)

    def _coord_channels(self, z: float, W: int, H: int) -> Tensor:
        """Sheets giving each pixel its x / y / z position, scaled 0 to 1."""
        if (W, H) not in self._xy_cache:
            self._xy_cache[(W, H)] = [torch.linspace(0, 1, H)[None, :].expand(W, H),
                                      torch.linspace(0, 1, W)[:, None].expand(W, H)]
        ramps = []
        if self.coords in ('xy', 'xyz'):
            ramps += self._xy_cache[(W, H)]
        if self.coords in ('z', 'xyz'):
            ramps.append(torch.full((W, H), z))
        if self.fourier_freqs:
            ramps = [f(math.pi * 2 ** k * r) for r in ramps
                     for k in range(self.fourier_freqs) for f in (torch.sin, torch.cos)]
        return torch.stack(ramps).float()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
        patient_id = img_path.stem.rsplit('_', 1)[0]
        pixel_spacing = self.spacing.get(patient_id)

        order, rank = self.vol[index]
        last = len(order) - 1
        z_norm: float = rank / last if last else 0.5
        window = [order[min(max(rank + d, 0), last)]
                  for d in range(-self.neighbours, self.neighbours + 1)]

        img: Tensor = torch.cat([self.img_transform(Image.open(self.files[i][0]))
                                 for i in window], dim=0)
        _, W, H = img.shape
        if self.coords != 'none':
            img = torch.cat([img, self._coord_channels(z_norm, W, H)], dim=0)
        assert img.shape[0] == self.n_channels

        data_dict = {"images": img,
                     "stems": img_path.stem,
                     "z_norm": z_norm}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
