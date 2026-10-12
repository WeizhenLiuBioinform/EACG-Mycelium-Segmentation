# MiT-B1 with the EACG coupled evidence/logit decoder.
custom_imports = dict(imports=['eacg'], allow_failed_imports=False)
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint = ('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
              'segformer/mit_b1_20220624-02e5a6a1.pth')

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=(512, 512)),
    backbone=dict(
        type='MixVisionTransformer', in_channels=3, embed_dims=64,
        num_stages=4, num_layers=[2, 2, 2, 2], num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='EACGHead', in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3], channels=256, dropout_ratio=0.1,
        num_classes=2, out_channels=2, ignore_index=255,
        norm_cfg=norm_cfg, align_corners=False,
        unc_guide=2, unc_guide_gamma=1.0,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False,
                 loss_weight=1.0, avg_non_ignore=True),
            dict(type='EACGUncertaintyCrossEntropyLoss', loss_weight=1.0,
                 avg_non_ignore=True, adaptive_weight=0, xi=0.0, tau=0.005,
                 adap_w_unk=1, adap_w_con=1, beta=1),
            dict(type='EACGUncertaintyBoundaryLoss', loss_weight=0.25,
                 loss_name='loss_ubl', avg_non_ignore=True,
                 adaptive_weight=0, xi=0.0, tau=0.005,
                 adap_w_unk=1, adap_w_con=1, beta=1,
                 # Scale signed distances by 0.1 before the boundary loss.
                 reward=True, discount=1.0, full_map_modify=0.1,
                 dist_kl=False, dist_kl_weight=0.0),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))
