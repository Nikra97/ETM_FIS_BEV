from collections import OrderedDict
import time
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner,)
from mmdet3d.utils import collect_env
from os import path as osp
import pickle
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmcv import Config
from timeit import default_timer as timer
import torch.utils.benchmark as benchmark
import torch.nn as nn
import torch
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmseg import __version__ as mmseg_version
import os
import mmcv 
from mmdet3d.utils import collect_env, get_root_logger
from os import path as osp

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet.datasets import (  # build_dataset,
    replace_ImageToTensor)
import gc 
from copy import deepcopy

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(False)


def update_cfg(
    cfg,
    n_future=4,
    receptive_field=3,
    resize_lim=(0.38, 0.55),
    final_dim=(256, 704),
    grid_conf={
        "xbound": [-51.2, 51.2, 0.8],
        "ybound": [-51.2, 51.2, 0.8],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
    det_grid_conf={
        "xbound": [-51.2, 51.2, 0.8],
        "ybound": [-51.2, 51.2, 0.8],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
    map_grid_conf={
        "xbound": [-30.0, 30.0, 0.15],
        "ybound": [-15.0, 15.0, 0.15],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
    motion_grid_conf={
        "xbound": [-50.0, 50.0, 0.5],
        "ybound": [-50.0, 50.0, 0.5],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
    t_input_shape=(128, 128),
    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
):

    cfg["det_grid_conf"] = det_grid_conf
    cfg["map_grid_conf"] = map_grid_conf
    cfg["motion_grid_conf"] = motion_grid_conf
    cfg["grid_conf"] = det_grid_conf

    cfg["model"]["temporal_model"]["input_shape"] = t_input_shape

    cfg["data"]["val"]["pipeline"][0]["data_aug_conf"]["resize_lim"] = resize_lim
    cfg["data"]["train"]["dataset"]["pipeline"][0]["data_aug_conf"][
        "resize_lim"
    ] = resize_lim
    cfg["data"]["val"]["pipeline"][0]["data_aug_conf"]["final_dim"] = final_dim
    cfg["data"]["train"]["dataset"]["pipeline"][0]["data_aug_conf"][
        "final_dim"
    ] = final_dim

    cfg["data"]["test"]["pipeline"][0]["data_aug_conf"]["resize_lim"] = resize_lim
    cfg["data"]["test"]["pipeline"][0]["data_aug_conf"]["final_dim"] = final_dim

    cfg["model"]["pts_bbox_head"]["cfg_motion"][
        "grid_conf"
    ] = motion_grid_conf  # motion_grid
    cfg["model"]["temporal_model"]["grid_conf"] = grid_conf
    cfg["model"]["transformer"]["grid_conf"] = grid_conf
    cfg["model"]["pts_bbox_head"]["grid_conf"] = grid_conf
    cfg["data"]["train"]["dataset"]["grid_conf"] = motion_grid_conf
    cfg["data"]["val"]["pipeline"][3]["grid_conf"] = motion_grid_conf
    cfg["data"]["test"]["pipeline"][3]["grid_conf"] = motion_grid_conf
    cfg["data"]["val"]["grid_conf"] = grid_conf
    cfg["data"]["test"]["grid_conf"] = grid_conf

    cfg["model"]["pts_bbox_head"]["det_grid_conf"] = det_grid_conf

    cfg["model"]["pts_bbox_head"]["map_grid_conf"] = map_grid_conf
    cfg["data"]["train"]["dataset"]["map_grid_conf"] = map_grid_conf
    cfg["data"]["test"]["pipeline"][2]["map_grid_conf"] = map_grid_conf
    cfg["data"]["val"]["pipeline"][2]["map_grid_conf"] = map_grid_conf
    cfg["data"]["test"]["map_grid_conf"] = map_grid_conf
    cfg["data"]["val"]["map_grid_conf"] = map_grid_conf

    cfg["model"]["pts_bbox_head"]["motion_grid_conf"] = motion_grid_conf

    cfg["data"]["test"]["pipeline"][5]["point_cloud_range"] = point_cloud_range
    cfg["data"]["train"]["pipeline"][5][
        "point_cloud_range"
    ] = point_cloud_range  # point_cloud_range=None
    cfg["data"]["train"]["pipeline"][6][
        "point_cloud_range"
    ] = point_cloud_range  # 'point_cloud_range =None
    cfg["data"]["val"]["pipeline"][5]["point_cloud_range"] = point_cloud_range

    cfg["model"]["pts_bbox_head"]["cfg_motion"]["receptive_field"] = receptive_field
    cfg["data"]["train"]["dataset"]["receptive_field"] = receptive_field
    cfg["model"]["temporal_model"]["receptive_field"] = receptive_field
    cfg["data"]["test"]["receptive_field"] = receptive_field
    cfg["data"]["val"]["receptive_field"] = receptive_field

    cfg["data"]["val"]["future_frames"] = n_future
    cfg["model"]["pts_bbox_head"]["cfg_motion"]["n_future"] = n_future
    cfg["data"]["test"]["future_frames"] = n_future
    cfg["data"]["train"]["dataset"]["future_frames"] = n_future

    cfg["data"]["train"]["dataset"]["pipeline"][4]["map_grid_conf"] = map_grid_conf
    cfg["data"]["train"]["dataset"]["pipeline"][5]["grid_conf"] = motion_grid_conf
    cfg["data"]["train"]["dataset"]["pipeline"][7]["point_cloud_range"] = point_cloud_range

    return cfg


def import_modules_load_config(cfg_file="beverse_tiny.py", samples_per_gpu=1):
    cfg_path = r"/home/niklas/ETM_BEV/BEVerse/projects/configs"
    cfg_path = os.path.join(cfg_path, cfg_file)

    cfg = Config.fromfile(cfg_path)

    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = cfg_path
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    return cfg


det_grid_conf = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 1.0],
}

motion_grid_conf = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 0.50],
}

map_grid_conf = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 1.0],
}

point_cloud_range_base = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range_extended_fustrum = [-62.0, -62.0, -5.0, 62.0, 62.0, 3.0]
#beverse_tiny_org motion_detr_tiny
cfg = import_modules_load_config(cfg_file="motion_detr_tiny.py")


# cfg = update_cfg(
#     cfg,
#     det_grid_conf=det_grid_conf,
#     grid_conf=det_grid_conf,
#     map_grid_conf=map_grid_conf,
#     motion_grid_conf=motion_grid_conf,
#     point_cloud_range=point_cloud_range_extended_fustrum,
#     #t_input_shape=(90, 155),
# )

cfg.data.train.dataset["data_root"] = '/home/niklas/ETM_BEV/BEVerse/data/nuscenes'
dataset = build_dataset(cfg.data.train)


# 3 5 time: 0.746, data_time: 0.042
data_loaders = [build_dataloader(
    dataset,
    samples_per_gpu=3,
    workers_per_gpu=6,
    dist=False,
    shuffle=False,)]


model = build_model(cfg.model, train_cfg=cfg.get(
    "train_cfg"), test_cfg=cfg.get('test_cfg'))
#model.init_weights()

# param_size = 0
# for param in model.parameters():
#     param_size += param.nelement() * param.element_size()
# buffer_size = 0
# for buffer in model.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (param_size + buffer_size) / 1024**2
# print('model size: {:.3f}MB'.format(size_all_mb))




cfg.checkpoint_config.meta = dict(
    mmdet_version=mmdet_version,
    mmseg_version=mmseg_version,
    mmdet3d_version=mmdet3d_version,
    config=cfg.pretty_text,
    CLASSES=dataset.CLASSES,
    PALETTE=dataset.PALETTE  # for segmentors
    if hasattr(dataset, 'PALETTE') else None)

# checkpoint_path = os.path.join(
#     "")


# checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')



load_model = False  
if load_model:
    # "temporal_model", "pts_bbox_head.task_decoders.motion", "pts_bbox_head.taskfeat_encoders.motion"]
    relevant_weights = ["img_backbone",
                        "transformer", "img_neck", "temporal_model", ]

    model_dict = model.state_dict()
    weights_tiny = torch.load(
        "/home/niklas/ETM_BEV/BEVerse/weights/beverse_tiny.pth")['state_dict']

    search_weights = tuple(weights_tiny.keys())

    new_weights = {}
    for k, v in model_dict.items():
        if k.startswith(tuple(relevant_weights)):
            new_weights[k] = weights_tiny[k].clone()
            print(
                f"Loaded weights for {k}, and required grad: {v.requires_grad}")
        else:
            print(f"ignored {k}")
    print("LOADED")
    weights_tiny = OrderedDict(new_weights)
    model_dict.update(weights_tiny)
    model.load_state_dict(model_dict)

    #print("Num of named gradients", len(list(model.named_parameters())))  # 367
    for (k, v) in (model.named_parameters()):
        if k.startswith(tuple(relevant_weights)):
            print(f"Turned off grads for {k}")
            v.requires_grad = False
        else:
            print(f"Grad enabled for {k}")

checkpoint_path = os.path.join(
    "/home/niklas/ETM_BEV/BEVerse/logs_cluster/segmentation_future_logs/epoch_10_correct_future_seg.pth")


checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')


# weights_tiny = torch.load( # 
#     "/home/niklas/ETM_BEV/BEVerse/weights/beverse_tiny.pth")["state_dict"]
# model.load_state_dict(weights_tiny)


model.cuda()
model = MMDataParallel(model, device_ids=[0])


log_path  = osp.abspath(r"/home/niklas/ETM_BEV/BEVerse/logs/testspeed")
mmcv.mkdir_or_exist(log_path)

# init the logger before other steps
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(
    log_path, f'{timestamp}.log')


logger_name = 'mmdet'
logger = get_root_logger(
    log_file=log_file, log_level=cfg.log_level, name=logger_name)


cfg.work_dir = "/home/niklas/ETM_BEV/BEVerse/logs/exp_logs/"
meta = dict()
# log env info
env_info_dict = collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +
            dash_line)
meta['env_info'] = env_info
meta['seed'] = 1337
meta['exp_name'] = "testname"

optimizer = build_optimizer(model, cfg.optimizer)

cfg.runner = {
    'type': 'EpochBasedRunner',
    'max_epochs': 1
}

# fp16_cfg = cfg.get('fp16', None)
# if fp16_cfg is not None:
#     print("")
#     wrap_fp16_model(model)
#     optimizer_config = Fp16OptimizerHook(
#         **cfg.optimizer_config, **fp16_cfg, distributed=False)
#else:
    # {'grad_clip': {'max_norm': 35, 'norm_type': 2}} ? 
optimizer_config = cfg.optimizer_config

# register hooks


runner = build_runner(
    cfg.runner,
    default_args=dict(
        model=model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta))

runner.register_training_hooks(cfg.lr_config, optimizer_config,
                               cfg.checkpoint_config, cfg.log_config,
                               cfg.get('momentum_config', None))

runner.run(data_loaders, cfg.workflow)
