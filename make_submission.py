#!/usr/bin/env python3
"""
Build the full submission folder in one go:

  group-XX/
      test/pred/Patient_41.nii.gz ...      model predictions on the test scans
      val/pred/Patient_XX.nii.gz ...       model predictions on the validation patients
      val/gt/Patient_XX.nii.gz ...         their ground truth (copied from the source data)
      group-XX.bundle                      git bundle of the repo
      bestmodel-group-XX.pkl               copy of the model

Steps:
  1. unzip test.zip (if needed)
  2. slice the test scans with the SAME preprocessing as training (reuses slice_segthor.slice_patient)
  3. predict val + test slices with the same code path (optional class weights)
  4. stitch slices back to full volumes in the original scan geometry (reuses stitch.merge_patient)
  5. copy val GT, run eval_volumetric.py -> val/<metric>.npz, check test volumes
  6. copy the model and create the git bundle
"""

import re
import sys
import json
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import torch
import numpy as np
import nibabel as nib
from PIL import Image
from torch.utils.data import DataLoader

from dataset import SliceDataset
from main import img_transforms, gt_transform   # same transforms as training
from slice_segthor import slice_patient, WINDOWS
from stitch import merge_patient

GROUP_RE = re.compile(r'(Patient_\d\d)_\d\d\d\d')


# ---------------------------------------------------------------- slicing
def slice_test(test_src: Path, dest: Path, args) -> None:
    img_dir = dest / 'test' / 'img'
    if img_dir.exists() and any(img_dir.glob('*.png')):
        print(f"Test slices already in {img_dir}, skipping slicing")
        return

    ids = sorted(Path(f.stem).stem for f in (test_src / 'test').glob('*.nii.gz'))
    assert ids, f"no test scans in {test_src / 'test'}"
    print(f"Slicing {len(ids)} test scans -> {dest / 'test'}")

    if dest.exists():
        shutil.rmtree(dest)
    pfun = partial(slice_patient, dest_path=dest / 'test', source_path=test_src,
                   shape=(256, 256), test_mode=True,
                   window=args.window, target_fov=args.target_fov)
    with Pool(max(1, args.procs)) as pool:
        pool.map(pfun, ids)


# ---------------------------------------------------------------- prediction
def predict(net, device, data_dir: Path, subset: str, out_dir: Path,
            weights: torch.Tensor, args) -> list[Path]:
    ds = SliceDataset(subset, data_dir,
                      img_transform=img_transforms[args.img_transform],
                      gt_transform=partial(gt_transform, args.K) if subset != 'test' else None,
                      neighbours=args.neighbours, coords=args.coords)
    assert len(ds) > 0, f"no slices in {data_dir / subset / 'img'}"
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(net(batch['images'].to(device)), dim=1).cpu()
            pred = (probs * weights[None, :, None, None]).argmax(dim=1)
            out = (pred * 63).numpy().astype(np.uint8)    # 0/63/126/189/252, as stitch.py expects
            for stem, arr in zip(batch['stems'], out):
                Image.fromarray(arr).save(out_dir / f"{stem}.png")

    return sorted(out_dir.glob('*.png'))


# ---------------------------------------------------------------- stitching
def stitch(pngs: list[Path], dest: Path, source_pattern: str, target_fov) -> list[str]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    groups: dict[str, list[int]] = {}
    for i, p in enumerate(pngs):
        groups.setdefault(GROUP_RE.match(p.stem).group(1), []).append(i)

    for pid in sorted(groups):
        merge_patient(pid, str(dest), pngs, groups[pid], 255, source_pattern, target_fov)
    print(f"Stitched {len(groups)} volumes -> {dest}")
    return sorted(groups)


# ---------------------------------------------------------------- checks
def val_metrics(val_dir: Path) -> None:
    """Run eval_volumetric.py on val/pred vs val/gt. It writes one <metric>.npz per metric
    (patient ID -> array of 4 organ values) into the pred folder; move them up to val/."""
    pred_dir, gt_dir = val_dir / 'pred', val_dir / 'gt'
    print("\n=== eval_volumetric.py on validation ===", flush=True)
    res = subprocess.run([sys.executable, 'eval_volumetric.py', str(pred_dir), str(gt_dir / '{id_}.nii.gz')],
                         check=True, capture_output=True, text=True)
    print(res.stdout)
    (val_dir / 'eval.txt').write_text(res.stdout)   # handy for the report; remove before archiving

    for old in val_dir.glob('*.npz'):
        old.unlink()
    for npz in sorted(pred_dir.glob('*.npz')):
        npz.rename(val_dir / npz.name)
    print(f"metrics: {[f.name for f in sorted(val_dir.glob('*.npz'))]}")


def check_test(pred_dir: Path, test_src: Path, ids: list[str]) -> None:
    print("\nTest volumes:")
    for pid in ids:
        arr = np.asarray(nib.load(pred_dir / f"{pid}.nii.gz").dataobj)
        scan = nib.load(test_src / 'test' / f"{pid}.nii.gz")
        labels = sorted(np.unique(arr).tolist())
        ok = labels == [0, 1, 2, 3, 4] and arr.shape == scan.shape
        print(f"  {pid}: shape {arr.shape}, labels {labels}" + ("" if ok else "  <-- CHECK"))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', required=True, help="e.g. group-11")
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data_dir', type=Path, required=True,
                   help="sliced training dataset, e.g. data/SEGTHOR_FULL_FOV_WIN (its val/ split is used)")
    p.add_argument('--source_dir', type=Path, required=True,
                   help="original NIfTI data the dataset was sliced from (train/Patient_XX/GT.nii.gz)")
    p.add_argument('--test_zip', type=Path, default=Path('data/test.zip'))
    p.add_argument('--test_src', type=Path, default=Path('data/segthor_test'))
    p.add_argument('--work_dir', type=Path, required=True, help="scratch space for slices")
    p.add_argument('--out_dir', type=Path, required=True, help="the submission folder is created inside this")
    p.add_argument('--class_weights', type=Path, default=None, help="weights.json from tune_weights.py")
    p.add_argument('--window', default='none', choices=['none', *WINDOWS.keys()])
    p.add_argument('--target_fov', type=float, default=None)
    p.add_argument('--K', type=int, default=5)
    p.add_argument('--img_transform', default='original', choices=list(img_transforms.keys()))
    p.add_argument('--neighbours', type=int, default=0)
    p.add_argument('--coords', default='none')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--procs', type=int, default=8)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--no_bundle', action='store_true')
    args = p.parse_args()
    print(args)

    sub = args.out_dir / args.group

    # 1. unzip
    if not (args.test_src / 'test').exists():
        print(f"Unzipping {args.test_zip} -> {args.test_src}")
        with zipfile.ZipFile(args.test_zip) as z:
            z.extractall(args.test_src)

    # 2. slice test
    test_sliced = args.work_dir / 'test_sliced'
    slice_test(args.test_src, test_sliced, args)

    # 3. model + weights
    device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")
    net = torch.load(args.model, map_location=device, weights_only=False).to(device)
    net.eval()

    weights = torch.ones(args.K)
    if args.class_weights is not None:
        weights = torch.tensor(json.loads(args.class_weights.read_text()), dtype=torch.float32)
        assert weights.numel() == args.K, weights
    print(f"class weights: {weights.tolist()}")

    # 4. validation: predict, stitch, copy GT, check
    val_pngs = predict(net, device, args.data_dir, 'val', args.work_dir / 'val_slices', weights, args)
    val_ids = stitch(val_pngs, sub / 'val' / 'pred',
                     str(args.source_dir / 'train' / '{id_}' / '{id_}.nii.gz'), args.target_fov)

    gt_dir = sub / 'val' / 'gt'
    gt_dir.mkdir(parents=True, exist_ok=True)
    for pid in val_ids:
        shutil.copy(args.source_dir / 'train' / pid / 'GT.nii.gz', gt_dir / f"{pid}.nii.gz")
    val_metrics(sub / 'val')

    # 5. test: predict, stitch, check
    test_pngs = predict(net, device, test_sliced, 'test', args.work_dir / 'test_slices', weights, args)
    test_ids = stitch(test_pngs, sub / 'test' / 'pred',
                      str(args.test_src / 'test' / '{id_}.nii.gz'), args.target_fov)
    check_test(sub / 'test' / 'pred', args.test_src, test_ids)

    # 6. model + bundle
    shutil.copy(args.model, sub / f"bestmodel-{args.group}.pkl")
    if not args.no_bundle:
        subprocess.run(['git', 'bundle', 'create', str((sub / f"{args.group}.bundle").resolve()), '--all'],
                       check=True)

    print(f"\nDone: {sub}")
    print(f"  val:  {len(val_ids)} pred + gt volumes, metrics in {sub / 'val'}")
    print(f"  test: {len(test_ids)} pred volumes")


if __name__ == '__main__':
    main()