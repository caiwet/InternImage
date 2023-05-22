# dataset settings
dataset_type = 'ETTDataset'
# classes = ('carina', 'tip', 'left clavicle', 'right clavicle')
classes = ('carina', 'tip', 'clavicles')

data_root = '/home/ec2-user/segmenter/ETTDATA/train_mimic_only/'
# data_root = '/home/ec2-user/segmenter/MAIDA/MIMIC_ETT_annotations/'
img_norm_cfg = dict(
    mean=[126.55846604, 126.55846604, 126.55846604], std=[55.47551373, 55.47551373, 55.47551373], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/train_annotations_enl5.json',
        img_prefix=data_root + 'images/train',
        # ann_file=data_root + 'annotations_enlarged_10.json',
        # img_prefix=data_root + 'enlarged10_1280/images/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val_annotations_enl5.json',
        img_prefix=data_root + 'images/val',
        # ann_file=data_root + 'annotations_enlarged_10.json',
        # img_prefix=data_root + 'enlarged10_1280/images/train',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        # ann_file=data_root + '../Test/downsized/RANZCR/annotations/test_annotations_enl5.json',
        # img_prefix=data_root + '../Test/downsized/RANZCR/images',
        ann_file=data_root + 'annotations/val_annotations_enl5.json',
        img_prefix=data_root + 'images/val',
        # ann_file=data_root + 'annotations_enlarged_10.json',
        # img_prefix=data_root + 'enlarged10_1280/images/train',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'], iou_thrs=[0.3], classwise=True)
