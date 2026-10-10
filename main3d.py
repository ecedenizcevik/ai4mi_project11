#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026

"""Training loop for the 3D U-Net.

Kept deliberately close to main.py: same optimizer, learning rate, loss and
logging, so a comparison against the 2D baseline varies the architecture and
the data layout only. Separate file because volume loading and slice loading
have little in common beyond the tensor names.
"""

import random
import argparse
from typing import Any
from pathlib import Path
from pprint import pprint

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from UNet3D import UNet3D
from losses import CrossEntropy
from dataset3d import VolumeDataset
from slice_segthor import get_splits
from utils import Dcm, probs2one_hot, tqdm_, dice_coef, dice_batch


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    K: int = 5
    net = UNet3D(1, K, kernels=args.kernels)
    net.init_weights()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Same split as slice_segthor.py produced for the 2D pipeline: seed 0, fold 0.
    # Reusing the function rather than hardcoding ids, so the two pipelines
    # cannot silently drift apart.
    random.seed(args.split_seed)
    train_ids, val_ids, _ = get_splits(args.data_dir, args.retains, args.fold)
    random.seed(args.seed)

    patch = tuple(args.patch)
    train_set = VolumeDataset(train_ids, args.data_dir, patch_size=patch,
                              patches_per_epoch=args.patches_per_epoch,
                              train=True, seed=args.seed)
    val_set = VolumeDataset(val_ids, args.data_dir, patch_size=patch,
                            patches_per_epoch=args.val_patches,
                            fg_ratio=0.0, train=False, seed=args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=0, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=0, shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return net, optimizer, device, train_loader, val_loader, K


def runTraining(args):
    print(f">>> Setting up 3D training on {args.data_dir}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    loss_fn = CrossEntropy(idk=list(range(K)))

    log_loss_tra = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val = torch.zeros((args.epochs, len(val_loader.dataset), K))
    # Patch-level 3D dice: pools voxels over the batch instead of averaging
    # per-patch ratios. Still not volume-level; that comes from inference.
    log_dice3d_val = torch.zeros((args.epochs, len(val_loader), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            if m == 'train':
                net.train()
                opt, cm, loader = optimizer, Dcm, train_loader
                log_loss, log_dice = log_loss_tra, log_dice_tra
                desc = f">> Training   ({e: 4d})"
            else:
                net.eval()
                opt, cm, loader = None, torch.no_grad, val_loader
                log_loss, log_dice = log_loss_val, log_dice_val
                desc = f">> Validation ({e: 4d})"

            with cm():
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:
                        opt.zero_grad()

                    assert 0 <= img.min() and img.max() <= 1
                    B = img.shape[0]

                    pred_logits = net(img)
                    pred_probs = F.softmax(pred_logits, dim=1)

                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)
                    if m == 'val':
                        log_dice3d_val[e, i, :] = dice_batch(pred_seg, gt)

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()

                    if opt:
                        loss.backward()
                        opt.step()

                    j += B
                    postfix = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                               "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    postfix |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                for k in range(1, K)}
                    tq_iter.set_postfix(postfix)

        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        np.save(args.dest / "dice3d_val.npy", log_dice3d_val)

        current_dice: float = log_dice3d_val[e, :, 1:].mean().item()
        print(f">>> epoch {e}: patch-pooled 3D dice {current_dice:05.3f}")
        if current_dice > best_dice:
            print(f">>> Improved: {best_dice:05.3f}->{current_dice:05.3f}")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(f"epoch {e}: {current_dice:05.3f}\n")
            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, default=Path("data/segthor_train_cleaned"))
    parser.add_argument('--patch', type=int, nargs=3, default=[128, 128, 32])
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--kernels', default=16, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--patches_per_epoch', default=250, type=int)
    parser.add_argument('--val_patches', default=100, type=int)
    parser.add_argument('--retains', default=5, type=int)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--split_seed', default=0, type=int,
                        help="Must match the seed slice_segthor.py used, or the "
                             "validation set will differ from the 2D baseline.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    pprint(args)
    runTraining(args)


if __name__ == '__main__':
    main()