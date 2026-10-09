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

        dims = (0, 2, 3)

        intersection = (pred * mask).sum(dim=dims)
        denominator = pred.sum(dim=dims) + mask.sum(dim=dims)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class TverskyLoss:
    """
    Multiclass Tversky loss.
    Alpha controls false-positive penalty, Beta controls the false-negative penalty
    When beta > alpha, loss penalizes missed organ voxels more strongly
    """

    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.alpha = kwargs.get('alpha', 0.3)
        self.beta = kwargs.get('beta', 0.7)
        self.smooth = kwargs.get('smooth', 1e-6)

        assert self.alpha >= 0
        assert self.beta >= 0
        assert self.alpha + self.beta > 0

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        mask = target[:, self.idk, ...].float()

        dims = (0, 2, 3)

        tp = (pred * mask).sum(dim=dims)
        fp = (pred * (1.0 - mask)).sum(dim=dims)
        fn = ((1.0 - pred) * mask).sum(dim=dims)

        tversky = (tp + self.smooth) / (
            tp
            + self.alpha * fp
            + self.beta * fn
            + self.smooth
        )

        present = mask.sum(dim=dims) > 0

        if not present.any():
            return pred_softmax.sum() * 0.0

        return 1.0 - tversky[present].mean()



class FocalLoss:
    """
    Multiclass Focal Loss for one-hot segmentation masks.

    gamma = 0 gives regular Cross-Entropy.
    gamma > 0 reduces the contribution of easy voxels.
    """

    def __init__(self, **kwargs):
        self.idk = kwargs["idk"]
        self.gamma = kwargs.get("gamma", 2.0)
        self.eps = kwargs.get("eps", 1e-10)

        assert self.gamma >= 0
        assert self.eps > 0

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        mask = target[:, self.idk, ...].float()
        log_p = pred.clamp_min(self.eps).log()
        focal_factor = (1.0 - pred).pow(self.gamma)
        loss = -mask * focal_factor * log_p

        return loss.sum() / (mask.sum() + self.eps)



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



class CrossEntropyTversky:
    """
    Combined Cross-Entropy and Tversky Loss.
    """

    def __init__(self, **kwargs):
        self.idk = kwargs["idk"]
        self.alpha = kwargs.get("alpha", 0.3)
        self.beta = kwargs.get("beta", 0.7)
        self.ce_weight = kwargs.get("ce_weight", 0.5)

        assert 0.0 <= self.ce_weight <= 1.0

        self.ce = CrossEntropy(
            idk=self.idk
        )

        self.tversky = TverskyLoss(
            idk=[k for k in self.idk if k != 0],
            alpha=self.alpha,
            beta=self.beta
        )

        print(
            f"Initialized {self.__class__.__name__} "
            f"with {kwargs}"
        )

    def __call__(self, pred_softmax, target):
        ce_loss = self.ce(pred_softmax, target)
        tversky_loss = self.tversky(pred_softmax, target)

        return (
            self.ce_weight * ce_loss
            + (1.0 - self.ce_weight) * tversky_loss
        )

