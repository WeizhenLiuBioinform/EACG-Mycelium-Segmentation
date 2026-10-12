# Copyright (c) 2025.
# EACG evidential loss.
import torch
import torch.nn as nn
import torch.distributions as D
from mmseg.registry import MODELS
from mmseg.models.losses.utils import weight_reduce_loss


def _one_hot(labels: torch.Tensor,
             num_classes: int,
             ignore_index: int):
    """Return one-hot labels [N,C,H,W] and a valid-pixel mask [N,H,W]."""

    N, H, W = labels.shape

    valid_mask = (labels >= 0) & (labels != ignore_index)

    y = labels.new_zeros((N, num_classes, H, W), dtype=torch.float32)

    if valid_mask.any():

        idx = labels.clone().long()

        idx[~valid_mask] = 0
        y.scatter_(1, idx.unsqueeze(1), 1.0)
        y *= valid_mask.unsqueeze(1).float()

    return y, valid_mask.float()


def _uce_per_pixel_per_class(alpha: torch.Tensor, y: torch.Tensor, class_weight=None):
    """Return unreduced, class-wise UCE terms with shape [N,C,H,W]."""

    S = alpha.sum(dim=1, keepdim=True)
    loss = y * (torch.digamma(S) - torch.digamma(alpha))
    if class_weight is not None:
        loss = loss * class_weight.view(1, -1, 1, 1)
    return loss


def entropy_reg(alpha, beta_reg=.001):
    """Return negative Dirichlet entropy scaled by beta_reg, as [N,1,H,W]."""
    alpha = alpha.permute(0, 2, 3, 1)

    reg = D.Dirichlet(alpha).entropy().unsqueeze(1)

    return -beta_reg * reg


@MODELS.register_module()
class EACGUncertaintyCrossEntropyLoss(nn.Module):
    """Evidential cross-entropy with negative Dirichlet entropy regularization.

    Inputs are Dirichlet concentrations, not classification logits.
    With mean reduction and no explicit avg_factor, masked losses are averaged
    over all pixels. avg_non_ignore does not change this denominator.
    adap_w_unk, adap_w_con and beta are accepted but do not affect the loss.
    """

    def __init__(self,
                 reduction='mean',
                 ignore_index=255,
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_uce',
                 avg_non_ignore=False,

                 adaptive_weight = 0,
                 xi = 1.0,
                 tau = 10.0,

                 adap_w_unk=1,
                 adap_w_con=1,
                 beta=1,
                 ):
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self.register_buffer('class_weight',
                             None if class_weight is None
                             else torch.tensor(class_weight, dtype=torch.float))

        self._loss_name = loss_name

        self.adap_w_unk = adap_w_unk
        self.adap_w_con = adap_w_con
        self.beta = beta

        self.xi = xi
        self.tau = tau
        if adaptive_weight not in (0, 1, 2):
            raise ValueError('adaptive_weight must be 0, 1, or 2.')
        self.adaptive_weight = adaptive_weight

        self.expects_alpha = True

    def forward(self,
                alpha,
                label,
                dist_map=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction = reduction_override or self.reduction
        C = alpha.size(1)

        y, valid_mask = _one_hot(label, C, kwargs.get('ignore_index', self.ignore_index))

        per_pix = _uce_per_pixel_per_class(alpha, y,
                                 class_weight=self.class_weight)

        if weight is None:
            weight = valid_mask
        else:
            weight = weight * valid_mask

        if self.adaptive_weight == 0:
            w = 1.0

        elif self.adaptive_weight == 1:

            dist_map_no_bg = dist_map[:, 1:, :, :]

            dist_map_adaptW = dist_map_no_bg.abs()

            ctrl_tau = 1 / self.tau

            dist_map_adaptW[dist_map_adaptW <= ctrl_tau] = 1.0
            dist_map_adaptW[dist_map_adaptW > ctrl_tau] = 0.0
            b_factor = dist_map_adaptW

            b_new = torch.cat([torch.zeros(b_factor.size(0), 1, b_factor.size(2), b_factor.size(3)).to(b_factor.device), b_factor], dim=1)

            S = alpha.sum(dim=1, keepdim=True)

            w = 1 + b_new * self.xi * alpha.size(1) / S

        elif self.adaptive_weight == 2:
            S = alpha.sum(dim=1, keepdim=True)
            w = 1 + self.xi * alpha.size(1) / S

        per_pix = w * per_pix
        per_pix = per_pix.sum(dim=1)
        # Add the negative Dirichlet entropy before masking and reduction.
        per_pix += entropy_reg(alpha, beta_reg=0.001).squeeze(1)

        loss = weight_reduce_loss(per_pix,
                                  weight=weight,
                                  reduction=reduction,
                                  avg_factor=avg_factor)
        return loss * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name
