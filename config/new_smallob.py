_base_ = '../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

custom_imports = dict(
    imports=['mine_aug.transforms'],
    allow_failed_imports=False
)

randomness = dict(seed=624, deterministic=False)

data_root = 'data/mine/'
class_name = ('rockbolt',)
num_classes = 1
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]
)
img_scale = (640, 640)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)
    )
)

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'

max_epochs = 24
close_crop_epochs = 10
switch_epoch = max_epochs - close_crop_epochs
affine_scale = 0.5
max_aspect_ratio = 100

pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'flip', 'flip_direction')
    )
]


train_pipeline = [
    *pre_transform,

    dict(
        type='MineClusterCropAug',
        prob=0.5,
        min_total_obj_num=15,
        dbscan_eps_ratio=0.12,
        dbscan_min_samples=3,
        min_cluster_size=4,
        core_k_range=(4, 8),
        crop_ratio_range=(0.45, 0.7),
        keep_area_ratio=0.50,
        max_try=20
    ),

    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114)
    ),

    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)
    ),

    *last_transform
]


train_pipeline_stage2 = [
    *pre_transform,

    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114)
    ),

    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)
    ),

    *last_transform
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param')
    )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_all_standard/train.json',
        data_prefix=dict(img='imgdir/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=4),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations_all_standard/test.json',
        data_prefix=dict(img='imgdir/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations_all_standard/test.json'
)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(lr=0.001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type='MultiStepLR', by_epoch=True, milestones=[18, 23], gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='auto',
        rule='greater'
    ),
    logger=dict(interval=20)
)

custom_hooks = [
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=switch_epoch,
        switch_pipeline=train_pipeline_stage2
    )
]

work_dir = './work_dirs/705_small'
resume = False