# dataset settings
dataset_type = 'CocoDataset'
data_root = 'D:\\datasets\\AI-TOD\\'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

metainfo = {
    'classes': ('airplane', 'bridge', 'storage-tank', 'ship',
                'swimming-pool', 'vehicle', 'person', 'wind-mill'),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
    ]
}

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/v1/aitod_trainval_v1.json',
        data_prefix=dict(img='trainval/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/v1/aitod_val_v1.json',
        data_prefix=dict(img='val/images'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='AITODMetric',
    ann_file=data_root + 'annotations/v1/aitod_val_v1.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/v1/aitod_test_v1.json',
        data_prefix=dict(img='test/images'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='AITODMetric',
    metric='bbox',
    format_only=False,
    ann_file=data_root + 'annotations/v1/aitod_test_v1.json',
    outfile_prefix='test/images')

