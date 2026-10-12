"""EACG configuration with boundary metrics at explicit dilation ratios."""
_base_ = ['./eacg_mit-b1_mycelium-512x512-160k.py']
custom_imports = dict(imports=['eacg', 'eacg.evaluation'], allow_failed_imports=False)
test_evaluator = dict(
    _delete_=True, type='EACGBoundaryMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore', 'assd', 'ece', 'biou_auc',
                 'fbiou001', 'fbiou005', 'fbiou02',
                 's2fbiou001', 's2fbiou005', 's2fbiou02'],
    dilation_ratio=0.02)
