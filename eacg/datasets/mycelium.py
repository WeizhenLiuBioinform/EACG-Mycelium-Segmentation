# MyceliumSeg dataset for EACG semantic segmentation.
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class EACGMyceliumDataset(BaseSegDataset):
    """MyceliumSeg semantic masks: 0 background, 1 mycelium, 255 ignored.

    Distance maps are computed after augmentation; no .npy files are needed.
    """

    METAINFO = dict(
        classes=('background', 'mycelium'),
        palette=[[0, 0, 0], [93, 191, 233]])

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix, **kwargs)
