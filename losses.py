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


from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss

class DiceLoss:
    """
    Soft Dice loss averaged over the selected classes.
    Intended primarily for foreground classes.
    """
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.smooth = kwargs.get('smooth', 1e-6)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        mask = target[:, self.idk, ...].float()

        # Aggregate over batch and spatial dimensions.
        dims = (0, 2, 3)

        intersection = (pred * mask).sum(dim=dims)
        denominator = pred.sum(dim=dims) + mask.sum(dim=dims)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class GeneralizedDiceLoss:
    """
    Generalized Dice loss.
    """
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.smooth = kwargs.get('smooth', 1e-6)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        mask = target[:, self.idk, ...].float()

        dims = (0, 2, 3)

        target_volume = mask.sum(dim=dims)
        pred_volume = pred.sum(dim=dims)
        intersection = (pred * mask).sum(dim=dims)

        weights = target_volume.new_zeros(target_volume.shape)
        present = target_volume > 0
        weights[present] = 1.0 / (target_volume[present] ** 2 + self.smooth)

        numerator = 2.0 * (weights * intersection).sum()
        denominator = (
            weights * (pred_volume + target_volume)
        ).sum()

        generalized_dice = (
            numerator + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - generalized_dice


class CrossEntropyDice:
    """
    Combined Cross-Entropy + foreground Soft Dice loss.
    """
    def __init__(self, **kwargs):
        self.ce = CrossEntropy(idk=kwargs['ce_idk'])
        self.dice = DiceLoss(idk=kwargs['dice_idk'])
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        ce_loss = self.ce(pred_softmax, target)
        dice_loss = self.dice(pred_softmax, target)

        return ce_loss + dice_loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
