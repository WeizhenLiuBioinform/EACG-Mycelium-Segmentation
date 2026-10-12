# Copyright (c) OpenMMLab. All rights reserved.
# Distance-map generation and packing for EACG training.
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.datasets.transforms import PackSegInputs
from mmseg.registry import TRANSFORMS
from scipy.ndimage import distance_transform_edt


@TRANSFORMS.register_module()
class EACGGenerateDistMap(BaseTransform):
    """Generate class-wise signed pixel distances after augmentation/padding.

    Inside a class: negative; outside: positive. Absent classes and ignored
    pixels have zero distances.
    """

    def __init__(self, num_classes=2, ignore_index=255):
        if num_classes < 2:
            raise ValueError('EACG requires background and at least one foreground class.')
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def transform(self, results):
        seg = results['gt_seg_map']
        if seg.ndim != 2:
            raise ValueError('Semantic masks must be single-channel class-index images.')
        ignore = seg == self.ignore_index
        invalid = ((seg < 0) | (seg >= self.num_classes)) & ~ignore
        if np.any(invalid):
            raise ValueError('Mask IDs must be 0..num_classes-1 or ignore_index.')
        distance = np.zeros((self.num_classes, *seg.shape), dtype=np.float32)
        for class_id in range(self.num_classes):
            mask = (seg == class_id) & ~ignore
            if mask.any():
                distance[class_id] = (distance_transform_edt(~mask)
                                      - distance_transform_edt(mask))
        distance[:, ignore] = 0.0
        results['gt_dist_map'] = distance
        return results


@TRANSFORMS.register_module()
class EACGPackSegInputs(PackSegInputs):
    """Use stock image/metadata packing and add the aligned distance maps."""

    def transform(self, results):
        packed = super().transform(results)
        if 'gt_dist_map' in results:
            distance = np.ascontiguousarray(results['gt_dist_map'], dtype=np.float32)
            if distance.ndim != 3:
                raise ValueError('gt_dist_map must have shape [C,H,W].')
            if distance.shape[1:] != results['gt_seg_map'].shape:
                raise ValueError('Distance maps and semantic masks must have matching sizes.')
            packed['data_samples'].gt_dist_map = PixelData(
                data=torch.from_numpy(distance))
        return packed
