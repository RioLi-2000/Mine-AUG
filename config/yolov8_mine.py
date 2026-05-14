_base_ = '../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

randomness = dict(
    seed=3407,
    deterministic=False,
    diff_rank_seed=False,
)
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


pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114)
    ),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'flip', 'flip_direction')
    )
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


train_cfg = dict(max_epochs=24, val_interval=1)

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


work_dir = './work_dirs/baseline'
resume = False
