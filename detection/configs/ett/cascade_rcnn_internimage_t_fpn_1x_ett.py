# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import wandb
_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/ett.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth'
model = dict(
    # backbone=dict(
    #     _delete_=True,
    #     type='InternImage',
    #     core_op='DCNv3',
    #     channels=64,
    #     depths=[4, 4, 18, 4],
    #     groups=[4, 8, 16, 32],
    #     mlp_ratio=4.,
    #     drop_path_rate=0.2,
    #     norm_layer='LN',
    #     layer_scale=1.0,
    #     offset_scale=1.0,
    #     post_norm=False,
    #     with_cp=False,
    #     out_indices=(0, 3),
    #     init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    backbone=dict(
        _delete_=True,
        type='CustomInternImage',
        core_op='DCNv3',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3, 4), # first 4 indices are internimage, last indice are gloria
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512, 1024], # first 4 channels are internimage, last channel are gloria
        out_channels=256,
        num_outs=5))
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
optimizer = dict(
    _delete_=True, type='AdamW', weight_decay=0.01,
    lr=0.000078,
    constructor='CustomLayerDecayOptimizerConstructor',
    # auto_scale_lr = dict(enable=True, base_batch_size=16),
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
                       depths=[4, 4, 18, 4]))
optimizer_config = dict(grad_clip=None)
# fp16 = dict(loss_scale=dict(init_scale=512))
max_epochs=12
num_last_epochs=3
evaluation = dict(save_best='auto',
                  interval=1,
                  dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
                  )
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,
    save_last=True,
)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

