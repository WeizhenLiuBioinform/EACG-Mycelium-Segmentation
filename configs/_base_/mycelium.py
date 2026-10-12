# Set EACG_DATA_ROOT before launching, or edit this default path.
data_root = '{{$EACG_DATA_ROOT:data/MyceliumSeg}}'
dataset_type = 'EACGMyceliumDataset'
crop_size = (512, 512)
image_scale = (4608, 3456)  # Training/evaluation resize scale (width, height).

# torchrun supplies WORLD_SIZE; a single-process launch defaults to 1.
global_batch_size = 8
world_size = int('{{$WORLD_SIZE:1}}')
if world_size < 1 or global_batch_size % world_size != 0:
    raise ValueError('WORLD_SIZE must be 1, 2, 4, or 8 for a total batch size of 8.')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=image_scale, ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=crop_size, pad_val=dict(img=0, seg=255)),
    dict(type='EACGGenerateDistMap', num_classes=2, ignore_index=255),
    dict(type='EACGPackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=global_batch_size // world_size,
    num_workers=4, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset', times=50,
        dataset=dict(
            type=dataset_type, data_root=data_root,
            img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=False,
            data_prefix=dict(img_path='train/image', seg_map_path='train/mask'),
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=False,
        data_prefix=dict(img_path='val/image', seg_map_path='val/mask'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=False,
        data_prefix=dict(img_path='test/image', seg_map_path='test/mask'),
        pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
