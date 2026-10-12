# Copyright (c) OpenMMLab. All rights reserved.
# EACG boundary metrics with explicit dilation-ratio suffixes.
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS

from .functional import (
    curve_sum,
    expected_calibration_error_alpha,
    img_average_symmetric_surface_distance,
    img_boundary_iou,
    img_boundary_iou_curve,
    img_fbiou,
    img_hausdorff_distance_95,
    img_s2fbiou,
    img_sbd,
)
from .functional import expected_calibration_error_logits


_AREA_METRICS = frozenset({'mIoU', 'mDice', 'mFscore'})
_LOGIT_METRICS = frozenset({'ece'})
_BOUNDARY_METRICS = frozenset({
    'assd',
    'hd95',
    'biou',
    'biou001',
    'biou005',
    'biou02',
    'biou_auc',
    'fbiou001',
    'fbiou005',
    'fbiou02',
    's2fbiou001',
    's2fbiou005',
    's2fbiou02',
    'sbd',
})
_SUPPORTED_METRICS = _AREA_METRICS | _LOGIT_METRICS | _BOUNDARY_METRICS
_BIOU_AUC_RATIOS = (0.001, 0.005, 0.01, 0.015, 0.02)
_NAMED_RATIOS = {
    'biou001': 0.001,
    'biou005': 0.005,
    'biou02': 0.02,
    'fbiou001': 0.001,
    'fbiou005': 0.005,
    'fbiou02': 0.02,
    's2fbiou001': 0.001,
    's2fbiou005': 0.005,
    's2fbiou02': 0.02,
}


class _BoundaryMetricAlignedBase(BaseMetric):
    """Shared implementation; subclasses select alpha or logits ECE."""

    ece_input = None

    def __init__(self,
                 ignore_index=255,
                 iou_metrics=('mIoU',),
                 nan_to_num=None,
                 beta=1,
                 collect_device='cpu',
                 output_dir=None,
                 format_only=False,
                 prefix=None,
                 dilation_ratio=0.02,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.metrics = list(iou_metrics)
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.dilation_ratio = float(dilation_ratio)

        metric_set = set(self.metrics)
        unsupported = metric_set - _SUPPORTED_METRICS
        if unsupported:
            raise KeyError(
                f'Unsupported metrics: {sorted(unsupported)}; '
                f'supported={sorted(_SUPPORTED_METRICS)}')
        if self.ece_input not in {'alpha', 'logits'}:
            raise ValueError(f'Invalid ece_input: {self.ece_input}')

        self._need_area = bool(metric_set & _AREA_METRICS)
        self._boundary_set = metric_set & (
            _BOUNDARY_METRICS | _LOGIT_METRICS)

        print_log(
            '[BoundaryMetric aligned v2] '
            f'ece_input={self.ece_input}, '
            f'need_area={self._need_area}, '
            f'boundary_in_process={sorted(self._boundary_set)}, '
            'ratio_contract=001:0.001,005:0.005,02:0.02',
            logger='current')

    def _compute_ece(self, seg_values, label):
        if self.ece_input == 'alpha':
            # F.one_hot cannot consume ignore_index, so replace ignored labels
            # before constructing one-hot targets, then mask them out.
            ignored = label == self.ignore_index
            clean_label = label.copy()
            clean_label[ignored] = 0
            _, _, ece = expected_calibration_error_alpha(
                seg_values, clean_label, exclude=ignored)
        else:
            _, _, ece = expected_calibration_error_logits(
                seg_values, label, ignore_index=self.ignore_index)
        return float(ece)

    def process(self, data_batch, data_samples):
        num_classes = len(self.dataset_meta['classes'])
        self.num_classes = num_classes

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()

            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)

                if self._need_area:
                    area = self.intersect_and_union(
                        pred_label, label, num_classes, self.ignore_index)
                else:
                    area = (None, None, None, None)

                pred_np = pred_label.cpu().numpy().astype(np.uint8)
                label_np = label.cpu().numpy().astype(np.uint8)
                ig = self.ignore_index
                boundary = {}

                if 'assd' in self._boundary_set:
                    boundary['assd'] = img_average_symmetric_surface_distance(
                        pred_np, label_np, num_classes, ig)
                if 'hd95' in self._boundary_set:
                    boundary['hd95'] = img_hausdorff_distance_95(
                        pred_np, label_np, num_classes, ig)

                if 'biou' in self._boundary_set:
                    boundary['biou'] = img_boundary_iou(
                        pred_np,
                        label_np,
                        num_classes,
                        ig,
                        dilation_ratio=self.dilation_ratio)

                for key in ('biou001', 'biou005', 'biou02'):
                    if key in self._boundary_set:
                        boundary[key] = img_boundary_iou(
                            pred_np,
                            label_np,
                            num_classes,
                            ig,
                            dilation_ratio=_NAMED_RATIOS[key])

                if 'biou_auc' in self._boundary_set:
                    boundary['biou_curve'] = img_boundary_iou_curve(
                        pred_np,
                        label_np,
                        num_classes,
                        ig,
                        dilation_ratios=_BIOU_AUC_RATIOS)

                for key in ('fbiou001', 'fbiou005', 'fbiou02'):
                    if key in self._boundary_set:
                        boundary[key] = img_fbiou(
                            pred_np,
                            label_np,
                            num_classes,
                            ig,
                            dilation_ratio=_NAMED_RATIOS[key])

                for key in ('s2fbiou001', 's2fbiou005', 's2fbiou02'):
                    if key in self._boundary_set:
                        boundary[key] = img_s2fbiou(
                            pred_np,
                            label_np,
                            num_classes,
                            ig,
                            dilation_ratio=_NAMED_RATIOS[key])

                if 'sbd' in self._boundary_set:
                    boundary['sbd'] = img_sbd(
                        pred_np, label_np, num_classes, ig, radius=1)

                if 'ece' in self._boundary_set:
                    seg_values = data_sample['seg_logits']['data'].squeeze()
                    ece = self._compute_ece(
                        seg_values.cpu().numpy(), label_np)
                    boundary['ece'] = np.full(
                        (num_classes,), ece, dtype=float)

                self.results.append((*area, boundary))

            if self.output_dir is not None:
                basename = osp.splitext(
                    osp.basename(data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                Image.fromarray(output_mask.astype(np.uint8)).save(
                    png_filename)

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        results = tuple(zip(*results))
        num_classes = self.num_classes
        ret_metrics = OrderedDict()

        if self._need_area:
            total_area_intersect = sum(results[0])
            total_area_union = sum(results[1])
            total_area_pred_label = sum(results[2])
            total_area_label = sum(results[3])
            ret_metrics['aAcc'] = (
                total_area_intersect.sum() / total_area_label.sum())

            for metric in self.metrics:
                if metric == 'mIoU':
                    ret_metrics['IoU'] = (
                        total_area_intersect / total_area_union)
                    ret_metrics['Acc'] = (
                        total_area_intersect / total_area_label)
                elif metric == 'mDice':
                    ret_metrics['Dice'] = (
                        2 * total_area_intersect /
                        (total_area_pred_label + total_area_label))
                    ret_metrics['Acc'] = (
                        total_area_intersect / total_area_label)
                elif metric == 'mFscore':
                    precision = (
                        total_area_intersect / total_area_pred_label)
                    recall = total_area_intersect / total_area_label
                    beta = self.beta
                    ret_metrics['Fscore'] = torch.tensor([
                        (1 + beta**2) * (p * r) /
                        ((beta**2 * p) + r)
                        for p, r in zip(precision, recall)
                    ])
                    ret_metrics['Precision'] = precision
                    ret_metrics['Recall'] = recall
        else:
            ret_metrics['aAcc'] = np.float64(0)

        boundary_dicts = list(results[4])

        def nanmean_vectors(key):
            vectors = [item[key] for item in boundary_dicts if key in item]
            if not vectors:
                return np.zeros((num_classes,), dtype=float)
            stacked = np.stack(vectors)
            with np.errstate(invalid='ignore'):
                sums = np.nansum(stacked, axis=0)
                counts = np.sum(~np.isnan(stacked), axis=0).astype(float)
                counts[counts == 0] = np.nan
                return sums / counts

        direct_keys = (
            'assd',
            'hd95',
            'biou',
            'biou001',
            'biou005',
            'biou02',
            'fbiou001',
            'fbiou005',
            'fbiou02',
            's2fbiou001',
            's2fbiou005',
            's2fbiou02',
            'sbd',
            'ece',
        )
        for key in direct_keys:
            if key in self._boundary_set:
                ret_metrics[key] = nanmean_vectors(key)

        if 'biou_auc' in self._boundary_set:
            curves = [
                item['biou_curve'] for item in boundary_dicts
                if 'biou_curve' in item
            ]
            if curves:
                with np.errstate(invalid='ignore'):
                    mean_curve = np.nanmean(np.stack(curves), axis=0)
                ratios = list(_BIOU_AUC_RATIOS)
                ret_metrics['biou001'] = mean_curve[:, ratios.index(0.001)]
                ret_metrics['biou005'] = mean_curve[:, ratios.index(0.005)]
                ret_metrics['biou02'] = mean_curve[:, ratios.index(0.02)]
                # biou_auc is the sum over the five ratios, not a trapezoidal integral.
                ret_metrics['biou_auc'] = curve_sum(mean_curve.T)

        ret_metrics = {
            key: value.numpy() if isinstance(value, torch.Tensor) else value
            for key, value in ret_metrics.items()
        }
        if self.nan_to_num is not None:
            ret_metrics = OrderedDict({
                key: np.nan_to_num(value, nan=self.nan_to_num)
                for key, value in ret_metrics.items()
            })

        class_names = self.dataset_meta['classes']
        ret_metrics_summary = OrderedDict({
            key: np.round(np.nanmean(value) * 100, 2)
            for key, value in ret_metrics.items()
        })
        metrics = {}
        for key, value in ret_metrics_summary.items():
            metrics[key] = value
            if key != 'aAcc':
                metrics['m' + key] = metrics.pop(key)

        ret_metrics_class = OrderedDict({
            key: np.round(value * 100, 2)
            for key, value in ret_metrics.items()
            if key != 'aAcc'
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        table = PrettyTable()
        for key, value in ret_metrics_class.items():
            table.add_column(key, value)
        print_log('per class results:', logger)
        print_log('\n' + table.get_string(), logger=logger)
        return metrics

    @staticmethod
    def intersect_and_union(pred_label, label, num_classes, ignore_index):
        mask = label != ignore_index
        pred_label = pred_label[mask]
        label = label[mask]
        intersect = pred_label[pred_label == label]
        intersect_cpu = intersect.to(torch.float32).cpu()
        pred_cpu = pred_label.to(torch.float32).cpu()
        label_cpu = label.to(torch.float32).cpu()
        area_intersect = torch.histc(
            intersect_cpu, bins=num_classes, min=0, max=num_classes - 1)
        area_pred_label = torch.histc(
            pred_cpu, bins=num_classes, min=0, max=num_classes - 1)
        area_label = torch.histc(
            label_cpu, bins=num_classes, min=0, max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label


@METRICS.register_module()
class EACGBoundaryMetric(_BoundaryMetricAlignedBase):
    """Aligned boundary metrics and top-1 ECE from segmentation logits."""

    ece_input = 'logits'
