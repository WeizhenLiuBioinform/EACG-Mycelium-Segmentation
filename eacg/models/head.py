# Copyright (c) OpenMMLab. All rights reserved.
# EACG coupled segmentation/evidence head.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils import resize

from mmseg.models.losses import accuracy
from torch import Tensor
from mmseg.utils import ConfigType, SampleList
from typing import List, Tuple

@MODELS.register_module()
class EACGHead(BaseDecodeHead):
    """EACG coupled segmentation/evidence head on SegFormer features.

    ``forward`` returns (logits, alpha); ``predict`` returns logits only.
    The default config uses unc_guide=2: U=1-max(alpha/sum(alpha)),
    followed by out + U*outs[0] + (1-U)*outs[-1].

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear',
                     unc_guide=0,
                     unc_guide_gamma=1.0,
                     **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        if self.num_classes < 2 or self.out_channels != self.num_classes:
            raise ValueError('EACG requires one evidence channel per class (C >= 2).')
        if unc_guide not in (0, 1, 2, 3, 4):
            raise ValueError('unc_guide must be 0, 1, 2, 3, or 4.')

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.conv_seg2 = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

        self.unc_guide = unc_guide
        self.unc_guide_gamma = unc_guide_gamma

        self.alpha_scale = 1.0
        self.beta_scale = 1.0

        if self.unc_guide == 2:
            self.identity = nn.Identity()

    def enn_seg(self, feat):
        """Return per-class evidence logits before ReLU and the unit prior."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg2(feat)
        return output

    def cal_uncertainty_aleatoric(self, alpha):
        """Return the mean-probability confidence proxy 1-max(alpha/sum(alpha))."""
        prob = alpha / alpha.sum(dim=1, keepdim=True)
        max_prob, _ = prob.max(dim=1, keepdim=True)
        u_alea = 1.0 - max_prob
        return u_alea

    def cal_uncertainty_epistemic(self, alpha):
        """Return Dirichlet vacuity C/sum(alpha)."""
        num_class = alpha.size(1)
        u_epis = num_class / alpha.sum(dim=1, keepdim=True)
        return u_epis

    def forward(self, inputs, limit=None):
        """Return classification logits and Dirichlet alpha at the first stage's size."""

        inputs = self._transform_inputs(inputs)
        outs = []

        for idx in range(len(inputs)):
            x = inputs[idx]

            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        evidence = self.enn_seg(out)
        evidence = evidence.relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1

        if self.unc_guide == 1:

            prob = alpha / alpha.sum(dim=1, keepdim=True)
            max_prob, _ = prob.max(dim=1, keepdim=True)
            U = 1.0 - max_prob

            out_refined = out * (1.0 + self.unc_guide_gamma * U)

            logits = self.cls_seg(out_refined)

        elif self.unc_guide == 2:

            # Gradients flow through the mean-probability uncertainty into alpha.
            prob = alpha / alpha.sum(dim=1, keepdim=True)
            max_prob, _ = prob.max(dim=1, keepdim=True)
            U = 1.0 - max_prob

            F_high = outs[0]

            F_low  = outs[-1]

            U_expand = U.expand_as(F_high)

            out_refined = (
                out
                + self.alpha_scale * (U_expand * F_high)
                + self.beta_scale  * ((1.0 - U_expand) * F_low)
            )

            out_refined = self.identity(out_refined)

            logits = self.cls_seg(out_refined)

        elif self.unc_guide == 3:

            prob = alpha / alpha.sum(dim=1, keepdim=True)
            max_prob, _ = prob.max(dim=1, keepdim=True)
            U = 1.0 - max_prob

            F_high = outs[0]

            F_low  = outs[-1]

            U_expand = U.expand_as(F_high)

            out_refined = (
                out
                + self.alpha_scale * ((1.0 - U_expand) * F_high)
                + self.beta_scale  * (U_expand * F_low)
            )

            logits = self.cls_seg(out_refined)

        elif self.unc_guide == 4:

            u_alea = self.cal_uncertainty_aleatoric(alpha)
            u_epis = self.cal_uncertainty_epistemic(alpha)

            F_high = outs[0]

            F_low  = outs[-1]

            u_alea = u_alea.expand_as(F_high).detach()
            u_epis = u_epis.expand_as(F_high).detach()

            out_refined = out\
                + (u_epis * (u_alea * F_high))\
                + (u_epis * ((1.0 - u_alea) * F_low))

            logits = self.cls_seg(out_refined)

        elif self.unc_guide == 0:
            logits = self.cls_seg(out)

        return logits, alpha

    def _stack_batch_dm(self, batch_data_samples: SampleList) -> Tensor:
        dm_semantic_segs = [
            data_sample.gt_dist_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(dm_semantic_segs, dim=0)

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Compute training losses from features, labels and distance maps.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): Samples containing
                gt_sem_seg and class-wise gt_dist_map.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, seg_alpha = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, seg_alpha, batch_data_samples)
        return losses

    def loss_by_feat(self, seg_logits: Tensor, seg_alpha: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): Classification logits with shape [N,C,h,w].
            seg_alpha (Tensor): Dirichlet concentrations with shape [N,C,h,w].
            batch_data_samples (List[:obj:`SegDataSample`]): Samples containing
                gt_sem_seg [1,H,W] and gt_dist_map [C,H,W].

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_distmap = self._stack_batch_dm(batch_data_samples)

        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_alpha = resize(
            input=seg_alpha,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None

        seg_label = seg_label.squeeze(1)
        if seg_distmap.shape != seg_alpha.shape:
            raise ValueError(
                'Distance maps must match resized alpha [N,C,H,W]. '
                'Generate them after all geometry transforms and padding.')

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            expects_alpha = getattr(loss_decode, 'expects_alpha', False)

            if expects_alpha:
                val = loss_decode(
                    seg_alpha,
                    seg_label,
                    seg_distmap,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                val = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = val
            else:
                loss[loss_decode.loss_name] += val

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (List[dict]): Image metadata containing img_shape
                and, when padding is applied, pad_shape.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, seg_alpha = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):

            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
