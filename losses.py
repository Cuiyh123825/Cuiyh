"""Loss functions for WBKMNet.

Re-uses the same DiceLoss formulation as KMUNet, adds two new boundary-
aware losses, and wraps them into a single composite loss with optional
deep supervision over a list of auxiliary predictions.

Losses
------
- ``DiceLoss``            : multi-class soft Dice loss.
- ``BoundaryDoULoss``     : Dice over the binary boundary region only,
                            extracted via morphological erosion/dilation.
- ``AdaptiveTverskyLoss`` : Tversky loss with skewed (alpha, beta) that
                            favours recall — useful for small-foreground
                            datasets such as BUSI.
- ``WBKMComboLoss``       : 0.5*(Dice+CE) + alpha*BoundaryDoU
                            + beta*AdaptiveTversky, with deep supervision
                            on a configurable set of auxiliary outputs.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Dice ------------------------------------

class DiceLoss(nn.Module):
    """Multi-class soft Dice loss (same convention as KMUNet)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1, H, W) integer labels
        tensors = []
        for i in range(self.n_classes):
            tensors.append((t == i * torch.ones_like(t)))
        return torch.cat(tensors, dim=1).float()

    @staticmethod
    def _dice(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = (score * target).sum()
        y_sum = (target * target).sum()
        z_sum = (score * score).sum()
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                weight: Optional[Sequence[float]] = None, softmax: bool = False) -> torch.Tensor:
        if softmax:
            logits = torch.softmax(logits, dim=1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = self._one_hot(target)
        if weight is None:
            weight = [1.0] * self.n_classes
        assert logits.size() == target.size(), \
            f"predict & target shape do not match: {logits.size()} vs {target.size()}"
        loss = 0.0
        for i in range(self.n_classes):
            loss = loss + self._dice(logits[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


# ----------------------------- Boundary-DoU ----------------------------

def _binary_boundary(mask: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """Extract the boundary band of a binary mask via dilate-minus-erode.

    Args:
        mask: ``(B, 1, H, W)`` float mask with values in ``[0, 1]``.
        ksize: kernel size used for the morphological ops.

    Returns:
        ``(B, 1, H, W)`` float boundary mask.
    """
    pad = ksize // 2
    dil = F.max_pool2d(mask, kernel_size=ksize, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size=ksize, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


class BoundaryDoULoss(nn.Module):
    """Boundary Dice-over-Union loss.

    Computes a soft-Dice between the predicted foreground probability and
    the GT mask restricted to the GT boundary region (a small band around
    the contour). This directly optimises boundary fidelity, which is
    where KMUNet under-performs the most.
    """

    def __init__(self, n_classes: int, fg_index: int = 1, ksize: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.fg_index = fg_index
        self.ksize = ksize

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                softmax: bool = True) -> torch.Tensor:
        if softmax:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        fg_target = (target == self.fg_index).float()
        boundary = _binary_boundary(fg_target, ksize=self.ksize)
        pred = probs[:, self.fg_index:self.fg_index + 1]

        # Restrict Dice to the boundary band so it is dominated by edge pixels.
        pred_b = pred * boundary
        targ_b = fg_target * boundary
        smooth = 1e-5
        inter = (pred_b * targ_b).sum()
        union = pred_b.sum() + targ_b.sum()
        return 1 - (2 * inter + smooth) / (union + smooth)


# ----------------------------- Adaptive Tversky ------------------------

class AdaptiveTverskyLoss(nn.Module):
    """Tversky loss with separate (alpha, beta) weights for FP / FN.

    Default ``alpha=0.3, beta=0.7`` skews the penalty toward false
    negatives — useful for sparse-foreground datasets like BUSI where the
    standard Dice tends to over-shrink the prediction.
    """

    def __init__(self, n_classes: int, alpha: float = 0.3, beta: float = 0.7,
                 fg_index: int = 1):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.fg_index = fg_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                softmax: bool = True) -> torch.Tensor:
        if softmax:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        fg_target = (target == self.fg_index).float()
        pred = probs[:, self.fg_index:self.fg_index + 1]

        tp = (pred * fg_target).sum()
        fp = (pred * (1 - fg_target)).sum()
        fn = ((1 - pred) * fg_target).sum()
        smooth = 1e-5
        tv = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        return 1 - tv


# ----------------------------- Composite loss --------------------------

class WBKMComboLoss(nn.Module):
    """Composite loss for WBKMNet training.

    L = 0.5*(Dice + CE) + alpha*BoundaryDoU + beta*AdaptiveTversky,
    with deep supervision: aux predictions are passed through the same
    loss and added with diminishing weights ``deep_sup_weights``.

    Forward usage:
        loss_fn = WBKMComboLoss(num_classes=2)
        # main: (B, C, H, W), aux: list of same-shape tensors or None
        loss, parts = loss_fn(main, aux, label)
    where ``label`` is an integer ``(B, H, W)`` tensor.
    """

    def __init__(
        self,
        num_classes: int,
        alpha_boundary: float = 0.3,
        beta_tversky: float = 0.3,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        deep_sup_weights: Sequence[float] = (0.5, 0.3, 0.2),
        fg_index: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_boundary = alpha_boundary
        self.beta_tversky = beta_tversky
        self.deep_sup_weights = list(deep_sup_weights)

        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.boundary = BoundaryDoULoss(num_classes, fg_index=fg_index)
        self.tversky = AdaptiveTverskyLoss(num_classes, alpha=tversky_alpha,
                                           beta=tversky_beta, fg_index=fg_index)

    def _single(self, logits: torch.Tensor, label_long: torch.Tensor,
                label_for_dice: torch.Tensor):
        soft = torch.softmax(logits, dim=1)
        l_ce = self.ce(logits, label_long)
        l_dice = self.dice(soft, label_for_dice)
        l_boundary = self.boundary(logits, label_for_dice, softmax=True)
        l_tversky = self.tversky(logits, label_for_dice, softmax=True)
        total = 0.5 * (l_dice + l_ce) \
                + self.alpha_boundary * l_boundary \
                + self.beta_tversky * l_tversky
        parts = {"ce": l_ce, "dice": l_dice, "boundary": l_boundary, "tversky": l_tversky}
        return total, parts

    def forward(
        self,
        main_logits: torch.Tensor,
        aux_logits: Optional[List[torch.Tensor]],
        label: torch.Tensor,
    ):
        label_long = label.long()
        label_for_dice = label.unsqueeze(1)

        loss_main, parts = self._single(main_logits, label_long, label_for_dice)
        total = loss_main
        if aux_logits is not None and len(aux_logits) > 0:
            weights = self.deep_sup_weights[: len(aux_logits)]
            for w, aux in zip(weights, aux_logits):
                aux_loss, _ = self._single(aux, label_long, label_for_dice)
                total = total + w * aux_loss
        parts["main"] = loss_main
        return total, parts


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 2, 64, 64
    main = torch.randn(B, C, H, W, requires_grad=True)
    aux = [torch.randn(B, C, H, W, requires_grad=True) for _ in range(3)]
    label = torch.randint(0, C, (B, H, W))
    loss_fn = WBKMComboLoss(num_classes=C)
    loss, parts = loss_fn(main, aux, label)
    loss.backward()
    print("loss:", float(loss),
          "| parts:", {k: float(v) for k, v in parts.items()})
