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


def scale_negative_part(dist_map: torch.Tensor, b: float = 2.0) -> torch.Tensor:
    """Multiply negative distances by b and leave nonnegative values unchanged.

    Accepts [N,C,H,W] or [C,H,W]. Returns a float32 tensor on the input device;
    a three-dimensional input gains a leading batch dimension.
    """
    if dist_map.ndim == 3:
        dist_map = dist_map.unsqueeze(0)

    out = dist_map.clone().float()

    neg_mask = out < 0
    if neg_mask.any():

        out[neg_mask] = out[neg_mask] * b

    return out


def _uce_per_pixel_per_class(alpha: torch.Tensor,
                   y: torch.Tensor,
                   dist_map: torch.Tensor,
                   class_weight=None,
                   pos_target_abs: float = 16.0,
                   neg_target_abs: float = 1.0,

                   reward: bool = False,

                   d0: float = 60.0,
                   eps: float = 1e-8,

                   adaptive_d0: bool = False,

                   discount: float = 1.0,

                   full_map_modify: float = 1.0,

                   dist_kl: bool = False,
                   dist_kl_weight: float = 0.0,

                   ):

    """Return unreduced boundary terms for foreground classes, as [N,C-1,H,W]."""

    dist_map_fg = dist_map[:, 1:, :, :]

    dist_map_fg = dist_map_fg * full_map_modify

    dist_map_balanced = scale_negative_part(dist_map_fg, b=0)

    y_cls_fg = y[:, 1:, :, :]

    alpha_cls_fg = alpha[:, 1:, :, :]
    alpha_cls_bg = alpha.sum(dim=1, keepdim=True) - alpha_cls_fg
    # Treat the concentration of competing classes as constant in this ratio.
    alpha_cls_bg_constant = alpha_cls_bg.detach()

    if reward:
        if discount == 0.0:
            L_fg = dist_map_fg * y_cls_fg * alpha_cls_fg / (alpha_cls_bg_constant + alpha_cls_fg)
        else:

            # The reward discount is constant with respect to the gradient.
            lambdax = torch.sigmoid(-discount * alpha_cls_fg / alpha_cls_bg).detach()

            L_fg = dist_map_fg * y_cls_fg * lambdax * alpha_cls_fg / (alpha_cls_bg_constant + alpha_cls_fg)

    loss = dist_map_balanced * alpha_cls_fg / (alpha_cls_fg + alpha_cls_bg_constant)

    if class_weight is not None:
        loss = loss * class_weight.view(1, -1, 1, 1)

    if reward:

        loss = loss + L_fg

    if dist_kl:

        w = signed_inverse_weight(dist_map_fg, normalize_io=True, tau=0.1, p=1.0)
        loss_bakl = dist_kl_weight * w * boundary_aware_entropy_reg(alpha_cls_fg)

        loss = loss + loss_bakl

    return loss


def boundary_aware_entropy_reg(alpha, beta_reg=.001):
    alpha = alpha.permute(0, 2, 3, 1)

    reg = D.Dirichlet(alpha).entropy().unsqueeze(1)

    return reg


def signed_inverse_weight(dist_map, normalize_io=True, tau=0.1, p=1.0, eps=1e-6):
    d = dist_map
    if normalize_io:
        d_in  = torch.clamp(-d, min=0.)
        d_out = torch.clamp( d, min=0.)
        dims = tuple(range(2, d.ndim))
        d_in_max  = d_in.amax(dim=dims, keepdim=True).clamp_min(eps)
        d_out_max = d_out.amax(dim=dims, keepdim=True).clamp_min(eps)

        abs_norm = d_in / d_in_max + d_out / d_out_max
        denom = abs_norm
    else:
        denom = d.abs()
    w = torch.sign(d) / (denom + tau)**p

    return w


@MODELS.register_module()
class EACGUncertaintyBoundaryLoss(nn.Module):
    """Evidential boundary suppression and detached-discount foreground reward.

    Channel 0 is background. class_weight, when supplied, contains C-1 weights
    for foreground classes. avg_non_ignore selects valid-pixel averaging.
    full_map_modify scales signed distances before computing boundary terms.
    adap_w_unk, adap_w_con, beta, pos_target_abs, neg_target_abs, d0 and
    adaptive_d0 are accepted but do not affect the loss.
    """

    def __init__(self,
                 reduction='mean',
                 ignore_index=255,
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ubl',
                 avg_non_ignore=False,

                 adaptive_weight = 0,
                 xi = 1.0,
                 tau = 10.0,

                 adap_w_unk=1,
                 adap_w_con=1,
                 beta=1,

                 pos_target_abs=16.0,
                 neg_target_abs=1.0,

                 reward=False,

                 d0=60.0,

                 adaptive_d0 = False,

                 discount = 1.0,

                 full_map_modify = 1.0,

                 dist_kl = False,
                 dist_kl_weight = 0.0,

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

        self.pos_target_abs = pos_target_abs
        self.neg_target_abs = neg_target_abs

        self.reward = reward
        self.d0 = d0

        self.adaptive_d0 = adaptive_d0

        self.discount = discount

        self.xi = xi
        self.tau = tau
        if adaptive_weight not in (0, 1, 2):
            raise ValueError('adaptive_weight must be 0, 1, or 2.')
        self.adaptive_weight = adaptive_weight

        self.full_map_modify = full_map_modify

        self.dist_kl = dist_kl
        self.dist_kl_weight = dist_kl_weight

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
        if dist_map is None or dist_map.shape != alpha.shape:
            raise ValueError('Boundary loss requires a distance map matching alpha [N,C,H,W].')
        if self.class_weight is not None and self.class_weight.numel() != C - 1:
            raise ValueError('Boundary class_weight must contain C-1 foreground weights.')

        target, valid_mask = _one_hot(label, C, kwargs.get('ignore_index', self.ignore_index))

        per_pix = _uce_per_pixel_per_class(alpha, target, dist_map,
                                 class_weight=self.class_weight,
                                 pos_target_abs = self.pos_target_abs,
                                 neg_target_abs = self.neg_target_abs,

                                 reward = self.reward,
                                 d0 = self.d0,

                                 adaptive_d0 = self.adaptive_d0,

                                 discount = self.discount,

                                 full_map_modify = self.full_map_modify,

                                 dist_kl = self.dist_kl,
                                 dist_kl_weight = self.dist_kl_weight
                                 )

        if weight is None:
            weight = valid_mask
        else:
            weight = weight * valid_mask

        if reduction == 'mean' and avg_factor is None and self.avg_non_ignore:
            avg_factor = max(valid_mask.sum().item(), 1.0)

        if self.adaptive_weight == 0:
            w = 1.0

        elif self.adaptive_weight == 1:

            dist_map_no_bg = dist_map[:, 1:, :, :]

            dist_map_adaptW = dist_map_no_bg.abs()

            ctrl_tau = 1 / self.tau

            dist_map_adaptW[dist_map_adaptW <= ctrl_tau] = 1.0
            dist_map_adaptW[dist_map_adaptW > ctrl_tau] = 0.0

            b_factor = dist_map_adaptW

            S = alpha.sum(dim=1, keepdim=True)

            w = 1 + b_factor * self.xi * alpha.size(1) / S

        elif self.adaptive_weight == 2:
            S = alpha.sum(dim=1, keepdim=True)
            w = 1 + self.xi * alpha.size(1) / S

        per_pix = w * per_pix
        per_pix = per_pix.sum(dim=1)

        loss = weight_reduce_loss(per_pix,
                                  weight=weight,
                                  reduction=reduction,
                                  avg_factor=avg_factor)
        return loss * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name
