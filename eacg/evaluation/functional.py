# Copyright (c) OpenMMLab. All rights reserved.
# EACG metric primitives; no registry side effects.
from typing import Optional, Tuple
import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import assd, hd95

def img_average_symmetric_surface_distance(pred_label,
                                           label,
                                           num_classes,
                                           ignore_index,
                                           label_map=dict(),
                                           reduce_zero_label=False):
    """Return per-class ASSD as an array with shape [num_classes]."""
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    h, w = pred_label.shape

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    valid = (label != ignore_index)

    assd_per_class = np.full((num_classes,), np.nan, dtype=float)
    for c in range(num_classes):
        gt_bin   = (label == c) & valid
        pred_bin = (pred_label == c) & valid
        assd_per_class[c] = safe_assd(gt_bin.astype(np.uint8), pred_bin.astype(np.uint8), connectivity=1)

    return assd_per_class


def img_hausdorff_distance_95(pred_label,
                              label,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    valid = (label != ignore_index)
    hd_per_class = np.full((num_classes,), np.nan, dtype=float)
    for c in range(num_classes):
        gt_bin   = (label == c) & valid
        pred_bin = (pred_label == c) & valid
        hd_per_class[c] = safe_hd95(gt_bin.astype(np.uint8), pred_bin.astype(np.uint8), connectivity=1)
    return hd_per_class


def img_boundary_iou(pred_label,
                     label,
                     num_classes,
                     ignore_index,
                     label_map=dict(),
                     reduce_zero_label=False,
                     dilation_ratio=0.02):
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    valid = (label != ignore_index)

    biou_per_class = np.full((num_classes,), np.nan, dtype=float)
    for c in range(num_classes):
        gt_bin   = ((label == c) & valid).astype(np.uint8)
        pred_bin = ((pred_label == c) & valid).astype(np.uint8)

        if gt_bin.sum() == 0 and pred_bin.sum() == 0:
            biou = np.nan
        elif gt_bin.sum() == 0 or pred_bin.sum() == 0:
            biou = 0.0
        else:
            biou = boundary_iou(gt_bin, pred_bin, dilation_ratio)
        biou_per_class[c] = biou
    return biou_per_class


def safe_assd(pred_bin, gt_bin, voxelspacing=None, connectivity=1):
    """Compute binary-mask ASSD; return NaN if either mask is empty."""
    if (pred_bin.sum() == 0) or (gt_bin.sum() == 0):
        return np.nan
    return assd(pred_bin, gt_bin, voxelspacing=voxelspacing, connectivity=connectivity)


def safe_hd95(pred_bin, gt_bin, voxelspacing=None, connectivity=1):
    if (pred_bin.sum() == 0) or (gt_bin.sum() == 0):
        return np.nan
    return hd95(pred_bin, gt_bin, voxelspacing=voxelspacing, connectivity=connectivity)


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """Compute inner-boundary IoU for two uint8 binary masks.

    The erosion width is max(1, round(dilation_ratio * image_diagonal)).
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt.astype(np.uint8), dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def mask_to_boundary(mask, dilation_ratio=0.02):
    """Return a uint8 inner-boundary mask using a square erosion kernel.

    The erosion width is max(1, round(dilation_ratio * image_diagonal)).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8, copy=False)

    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))

    if dilation < 1:
        dilation = 1

    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]

    return mask - mask_erode


def bin_predictions(y_score, y_true, n_bins=10):
    max_prob, max_ind = y_score.max(dim=1)

    acc_binned = torch.zeros((n_bins,))
    conf_binned = torch.zeros((n_bins,))
    bin_cardinalities = torch.zeros((n_bins,))

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    lower_bin_boundary = bin_boundaries[:-1]
    upper_bin_boundary = bin_boundaries[1:]

    corrects = max_ind == y_true.argmax(dim=1)

    for b in range(n_bins):
        in_bin = (max_prob < upper_bin_boundary[b]) & (max_prob >= lower_bin_boundary[b])
        bin_cardinality = in_bin.sum()
        bin_cardinalities[b] = bin_cardinality

        if bin_cardinality > 0:
            acc_binned[b] = corrects[in_bin].float().mean()
            conf_binned[b] = max_prob[in_bin].mean()

    return acc_binned, conf_binned, bin_cardinalities


def expected_calibration_error_alpha(alpha, label,
                                     n_bins: int = 10,
                                     exclude=None,
                                     eps: float = 1e-6):
    """Compute top-1 ECE from channel-first Dirichlet concentrations.

    alpha has shape [N,C,H,W] or [C,H,W]; label has shape [N,H,W] or [H,W].
    All labels must be valid class IDs before one-hot encoding. To ignore
    pixels, replace their IDs with a valid ID and supply the exclude mask.
    Returns per-bin confidence, per-bin accuracy and scalar ECE; n_bins=10.
    """
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    label = torch.as_tensor(label, dtype=torch.int64)

    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(0)
    elif alpha.ndim == 4 and alpha.shape[-1] != alpha.shape[1]:

        pass

    if label.ndim == 2:
        label = label.unsqueeze(0)

    N, C, H, W = alpha.shape

    S = alpha.sum(dim=1, keepdim=True) + eps
    prob = alpha / S

    label_oh = F.one_hot(label, num_classes=C)
    label_oh = label_oh.permute(0,3,1,2).float()

    if exclude is not None:
        excl = torch.as_tensor(exclude).bool()
        if excl.ndim == 2: excl = excl.unsqueeze(0)
        prob     = prob.masked_fill(excl.unsqueeze(1), 0.)
        label_oh = label_oh.masked_fill(excl.unsqueeze(1), 0.)

    y_score = prob.permute(0,2,3,1).reshape(-1, C)
    y_true  = label_oh.permute(0,2,3,1).reshape(-1, C)

    keep = y_true.sum(dim=1) > 0
    y_score = y_score[keep]
    y_true  = y_true[keep]

    acc, conf, bin_card = bin_predictions(y_score, y_true, n_bins)
    ece = (torch.abs(acc - conf) * bin_card).sum() / y_true.shape[0]
    return conf, acc, ece


def _ratio_to_dilations(h, w, dilation_ratios):
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dils = [int(round(r * img_diag)) for r in dilation_ratios]
    dils = [max(1, d) for d in dils]
    return dils


def _dist_transform_inside(mask_uint8):
    """Return the inside-mask chessboard distance to a zero-valued pixel."""
    h, w = mask_uint8.shape
    padded = cv2.copyMakeBorder(mask_uint8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, distanceType=cv2.DIST_C, maskSize=3)
    dist = dist[1:h+1, 1:w+1]
    return dist


def boundary_iou_curve(gt, pred, dilation_ratios):
    """Return K boundary-IoU values for the supplied dilation ratios.

    gt and pred are uint8 binary masks with shape [H,W] and values in {0,1}.
    """
    gt = gt.astype(np.uint8, copy=False)
    pred = pred.astype(np.uint8, copy=False)
    h, w = gt.shape
    dils = _ratio_to_dilations(h, w, dilation_ratios)

    gt_dist = _dist_transform_inside(gt)
    pr_dist = _dist_transform_inside(pred)

    out = np.zeros((len(dils),), dtype=float)
    for i, d in enumerate(dils):
        gt_b = (gt > 0) & (gt_dist <= d)
        pr_b = (pred > 0) & (pr_dist <= d)

        inter = np.logical_and(gt_b, pr_b).sum()
        union = np.logical_or(gt_b, pr_b).sum()
        out[i] = (inter / union) if union > 0 else np.nan
    return out


def curve_sum(curve, weights=None):
    """Sum K curve samples, preserving NaN for entirely missing classes.

    curve has shape [K,C] or [K]. Optional weights have shape [K]. Returns
    an array of shape [C], or a scalar for a single class.
    """
    curve = np.asarray(curve, dtype=float)

    if curve.ndim == 1:
        curve = curve[:, None]

    K, C = curve.shape

    if weights is None:
        s = np.nansum(curve, axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        assert w.shape[0] == K
        s = np.nansum(curve * w[:, None], axis=0)

    all_nan = np.all(np.isnan(curve), axis=0)
    s = s.astype(float)
    s[all_nan] = np.nan
    return s if C > 1 else float(s[0])


def img_boundary_iou_curve(pred_label,
                           label,
                           num_classes,
                           ignore_index,
                           label_map=dict(),
                           reduce_zero_label=False,
                           dilation_ratios=(0.001, 0.002, 0.005, 0.01, 0.02)):

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    valid = (label != ignore_index)
    K = len(dilation_ratios)
    biou = np.full((num_classes, K), np.nan, dtype=float)

    present = np.unique(np.concatenate([label[valid].reshape(-1), pred_label[valid].reshape(-1)]))
    present = present[(present >= 0) & (present < num_classes)]

    for c in present:
        gt_bin   = ((label == c) & valid).astype(np.uint8)
        pred_bin = ((pred_label == c) & valid).astype(np.uint8)

        if gt_bin.sum() == 0 and pred_bin.sum() == 0:
            continue
        elif gt_bin.sum() == 0 or pred_bin.sum() == 0:
            biou[c, :] = 0.0
        else:
            biou[c, :] = boundary_iou_curve(gt_bin, pred_bin, dilation_ratios)

    return biou


def _ratio_to_d(h, w, dilation_ratio):
    diag = (h * h + w * w) ** 0.5
    d = int(round(float(dilation_ratio) * diag))
    return max(1, d)


def mask_to_full_band(mask: np.ndarray,
                      dilation_ratio: float = 0.02,
                      valid_mask: np.ndarray = None,
                      dist_type=cv2.DIST_C):
    """Return the union of the inner and outer boundary bands as uint8.

    Each band contains pixels within d of the opposite mask value, where
    d=max(1, round(dilation_ratio * image_diagonal)). The default distance
    is chessboard distance. valid_mask, if supplied, clips the resulting band.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.astype(np.uint8, copy=False)
    h, w = mask.shape
    d = _ratio_to_d(h, w, dilation_ratio)

    pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    dist_in = cv2.distanceTransform(pad, dist_type, 3)[1:h+1, 1:w+1]
    inner = (mask > 0) & (dist_in <= d)

    inv = (1 - pad).astype(np.uint8, copy=False)
    dist_out = cv2.distanceTransform(inv, dist_type, 3)[1:h+1, 1:w+1]
    outer = (mask == 0) & (dist_out <= d)

    band = inner | outer

    if valid_mask is not None:
        band = band & valid_mask

    return band.astype(np.uint8)


def full_band_iou(gt_bin: np.ndarray,
                  pred_bin: np.ndarray,
                  dilation_ratio: float = 0.02,
                  valid_mask: np.ndarray = None):
    gt_band = mask_to_full_band(gt_bin, dilation_ratio, valid_mask=valid_mask)
    pr_band = mask_to_full_band(pred_bin, dilation_ratio, valid_mask=valid_mask)

    inter = np.logical_and(gt_band > 0, pr_band > 0).sum()
    union = np.logical_or(gt_band > 0, pr_band > 0).sum()
    return (inter / union) if union > 0 else np.nan


def solidity_ratio(mask: np.ndarray, eps: float = 1e-6):
    """Return foreground area divided by rasterized convex-hull area plus eps.

    Empty masks or masks without external contours return NaN.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = (mask > 0).astype(np.uint8, copy=False)

    area = float(mask.sum())
    if area <= 0:
        return np.nan

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan

    pts = np.vstack(contours)
    hull = cv2.convexHull(pts)

    hull_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    hull_area = float(hull_mask.sum())

    return area / (hull_area + eps)


def s2fb_iou(gt_bin: np.ndarray,
             pred_bin: np.ndarray,
             dilation_ratio: float = 0.02,
             valid_mask: np.ndarray = None,
             eps: float = 1e-6):
    fb = full_band_iou(gt_bin, pred_bin, dilation_ratio, valid_mask=valid_mask)
    if np.isnan(fb):
        return np.nan

    sol_g = solidity_ratio(gt_bin, eps=eps)
    sol_s = solidity_ratio(pred_bin, eps=eps)

    if np.isnan(sol_g) or np.isnan(sol_s):

        return float(fb)

    base = (sol_g * sol_s) / (sol_g + sol_s + eps)
    base = max(float(base), eps)

    return float(fb) * (base ** (1.0 - float(fb)))


def img_fbiou(pred_label, label, num_classes, ignore_index,
              label_map=dict(), reduce_zero_label=False,
              dilation_ratio=0.02):
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    valid = (label != ignore_index)
    out = np.full((num_classes,), np.nan, dtype=float)

    present = np.unique(np.concatenate([label[valid].reshape(-1), pred_label[valid].reshape(-1)]))
    present = present[(present >= 0) & (present < num_classes)]

    for c in present:
        gt_bin   = ((label == c) & valid).astype(np.uint8)
        pr_bin   = ((pred_label == c) & valid).astype(np.uint8)

        if gt_bin.sum() == 0 and pr_bin.sum() == 0:
            continue
        elif gt_bin.sum() == 0 or pr_bin.sum() == 0:
            out[c] = 0.0
        else:
            out[c] = full_band_iou(gt_bin, pr_bin, dilation_ratio, valid_mask=valid)

    return out


def img_s2fbiou(pred_label, label, num_classes, ignore_index,
                label_map=dict(), reduce_zero_label=False,
                dilation_ratio=0.02):
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    valid = (label != ignore_index)
    out = np.full((num_classes,), np.nan, dtype=float)

    present = np.unique(np.concatenate([label[valid].reshape(-1), pred_label[valid].reshape(-1)]))
    present = present[(present >= 0) & (present < num_classes)]

    for c in present:
        gt_bin   = ((label == c) & valid).astype(np.uint8)
        pr_bin   = ((pred_label == c) & valid).astype(np.uint8)

        if gt_bin.sum() == 0 and pr_bin.sum() == 0:
            continue
        elif gt_bin.sum() == 0 or pr_bin.sum() == 0:
            out[c] = 0.0
        else:
            out[c] = s2fb_iou(gt_bin, pr_bin, dilation_ratio, valid_mask=valid)

    return out


def mask_to_boundary_band(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Return a uint8 inner band obtained by subtracting the eroded mask.

    Uses a 3x3 square kernel with max(1, radius) erosion iterations and
    OpenCV's default erosion border handling.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = (mask > 0).astype(np.uint8, copy=False)
    if radius <= 0:
        radius = 1
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(mask, kernel, iterations=radius)
    band = mask - er
    return (band > 0).astype(np.uint8)


def box_sum(img_u8: np.ndarray, radius: int) -> np.ndarray:
    """Return unnormalized pixel sums over square neighborhoods of side 2r+1."""
    k = 2 * int(radius) + 1
    return cv2.boxFilter(
        img_u8.astype(np.float32),
        ddepth=-1,
        ksize=(k, k),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT
    )


def sbd_binary(gt: np.ndarray, pred: np.ndarray,
               radius: int = 1,
               valid_mask: np.ndarray = None,
               eps: float = 1e-12) -> float:
    """Average local Dice over both masks' valid boundary pixels.

    gt and pred are binary masks of shape [H,W]. valid_mask excludes ignored
    pixels. Returns NaN when both input masks are empty, and zero when no
    valid boundary pixels remain.
    """
    gt = (gt > 0).astype(np.uint8, copy=False)
    pred = (pred > 0).astype(np.uint8, copy=False)

    if gt.sum() == 0 and pred.sum() == 0:
        return np.nan

    if valid_mask is None:
        valid_mask = np.ones_like(gt, dtype=np.uint8)
    else:
        valid_mask = (valid_mask > 0).astype(np.uint8, copy=False)

    kernel = np.ones((3, 3), np.uint8)
    valid_core = cv2.erode(valid_mask, kernel, iterations=max(1, radius))

    gt = gt * valid_mask
    pred = pred * valid_mask

    b_gt = mask_to_boundary_band(gt, radius=radius) * valid_core
    b_pr = mask_to_boundary_band(pred, radius=radius) * valid_core

    n_gt = float(b_gt.sum())
    n_pr = float(b_pr.sum())
    denom = n_gt + n_pr
    if denom <= 0:

        return 0.0

    inter = (gt & pred).astype(np.uint8)
    a = box_sum(gt, radius)
    b = box_sum(pred, radius)
    i = box_sum(inter, radius)

    local_dice = (2.0 * i) / (a + b + eps)

    score = float(local_dice[b_gt > 0].sum() + local_dice[b_pr > 0].sum())
    return score / denom


def img_sbd(pred_label, label, num_classes, ignore_index,
            label_map=dict(), reduce_zero_label=False,
            radius: int = 1):
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)
    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    valid = (label != ignore_index).astype(np.uint8)

    out = np.full((num_classes,), np.nan, dtype=float)

    present = np.unique(np.concatenate([label[valid > 0].reshape(-1),
                                        pred_label[valid > 0].reshape(-1)]))
    present = present[(present >= 0) & (present < num_classes)]

    for c in present:
        gt_bin = ((label == c) & (valid > 0)).astype(np.uint8)
        pr_bin = ((pred_label == c) & (valid > 0)).astype(np.uint8)
        out[c] = sbd_binary(gt_bin, pr_bin, radius=radius, valid_mask=valid)

    return out


@torch.no_grad()
def bin_predictions_top1(y_prob: torch.Tensor,
                         y_true_idx: torch.Tensor,
                         n_bins: int = 10
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bin top-1 predictions by confidence.

    Args:
        y_prob: Class probabilities with shape [Npix,C].
        y_true_idx: Integer ground-truth class IDs with shape [Npix].
        n_bins: Number of equal-width bins. Defaults to 10.

    Returns:
        Tuple of mean confidence, mean accuracy and pixel count per bin,
        each with shape [n_bins].
    """
    device = y_prob.device
    conf, pred = torch.max(y_prob, dim=1)
    correct = (pred == y_true_idx).float()

    bin_id = torch.clamp((conf * n_bins).long(), max=n_bins - 1)

    bin_count = torch.bincount(bin_id, minlength=n_bins).to(device).float()

    bin_conf_sum = torch.zeros(n_bins, device=device).index_add(0, bin_id, conf)
    bin_acc_sum  = torch.zeros(n_bins, device=device).index_add(0, bin_id, correct)

    denom = torch.clamp(bin_count, min=1.0)
    bin_conf = bin_conf_sum / denom
    bin_acc  = bin_acc_sum  / denom
    return bin_conf, bin_acc, bin_count


@torch.no_grad()
def expected_calibration_error_logits(logits,
                                      label,
                                      n_bins: int = 10,
                                      exclude: Optional[torch.Tensor] = None,
                                      ignore_index: Optional[int] = None,
                                      from_logits: bool = True,
                                      eps: float = 1e-12):
    """Compute top-1 ECE from channel-first logits or probabilities.

    Args:
        logits: Tensor or array with shape [N,C,H,W] or [C,H,W].
        label: Integer class IDs with shape [N,H,W] or [H,W].
        n_bins: Number of equal-width bins. Defaults to 10.
        exclude: Optional boolean mask [N,H,W] or [H,W]; True excludes a pixel.
        ignore_index: Optional label value to exclude.
        from_logits: Apply softmax if True; otherwise accept probabilities.
        eps: Small value added to the probability-normalization denominator.

    Returns:
        Per-bin mean confidence, per-bin mean accuracy and scalar ECE on CPU.
        Empty valid-pixel sets return zeros.
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    label  = torch.as_tensor(label,  dtype=torch.int64)

    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    elif logits.ndim == 4 and logits.shape[1] < logits.shape[-1]:

        pass

    if label.ndim == 2:
        label = label.unsqueeze(0)
    elif label.ndim == 3 and label.size(0) == 1 and label.shape[0] != logits.shape[0]:

        pass

    N, C, H, W = logits.shape

    prob = F.softmax(logits, dim=1) if from_logits else torch.clamp(logits, min=0.0, max=1.0)

    prob = prob / (prob.sum(dim=1, keepdim=True) + eps)

    valid = torch.ones((N, H, W), dtype=torch.bool, device=prob.device)
    if exclude is not None:
        excl = torch.as_tensor(exclude, dtype=torch.bool, device=prob.device)
        if excl.ndim == 2:
            excl = excl.unsqueeze(0)
        valid &= ~excl
    if ignore_index is not None:
        valid &= (label != ignore_index)

    y_prob = prob.permute(0, 2, 3, 1).reshape(-1, C)
    y_true = label.reshape(-1)
    vmask  = valid.reshape(-1)
    y_prob = y_prob[vmask]
    y_true = y_true[vmask]

    if y_true.numel() == 0:
        nb = torch.zeros(n_bins, dtype=torch.float32)
        return nb, nb, torch.tensor(0.0, dtype=torch.float32)

    bin_conf, bin_acc, bin_count = bin_predictions_top1(y_prob, y_true, n_bins=n_bins)
    ece = (torch.abs(bin_acc - bin_conf) * (bin_count / bin_count.sum().clamp(min=1.0))).sum()
    return bin_conf.cpu(), bin_acc.cpu(), ece.cpu()
