from mmseg import __version__ as mmseg_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet import __version__ as mmdet_version
import pickle
from os import path as osp
from mmdet3d.utils import collect_env
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
import os
from custome_logger import setup_custom_logger

logger = setup_custom_logger()
logger.debug("test")

import torch


from mmcv import Config
from mmcv.runner import wrap_fp16_model
from mmdet3d.models import build_model

from mmdet3d.datasets import build_dataloader,build_dataset
from mmcv.parallel import MMDataParallel
from mmdet.datasets import ( #build_dataset,
                            replace_ImageToTensor)

def update_cfg(
    cfg,
    n_future=4,
    receptive_field=4,
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
    ] = point_cloud_range  #'point_cloud_range =None
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
    cfg["data"]["train"]["dataset"]["pipeline"][7][
        "point_cloud_range"
    ] = point_cloud_range

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
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
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


torch.backends.cudnn.benchmark = True

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

#model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))

train_setup = True  
if train_setup:
    cfg.data.train.dataset["data_root"] = '/home/niklas/ETM_BEV/BEVerse/data/nuscenes'
    dataset = build_dataset(cfg.data.train)
else:
    cfg.data.test["data_root"] = '/home/niklas/ETM_BEV/BEVerse/data/nuscenes'
    dataset = build_dataset(cfg.data.test)
data_loaders = [build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,)]

sample = next(iter(data_loaders[0]))


model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get('test_cfg'))
#wrap_fp16_model(model)

load_model = False 
if load_model:
    cfg.checkpoint_config.meta = dict(
        mmdet_version=mmdet_version,
        mmseg_version=mmseg_version,
        mmdet3d_version=mmdet3d_version,
        config=cfg.pretty_text,
        CLASSES=dataset.CLASSES,
        PALETTE=dataset.PALETTE  # for segmentors
        if hasattr(dataset, 'PALETTE') else None)

    weights_tiny = torch.load(
        "/home/niklas/ETM_BEV/BEVerse/logs_cluster/epoch_5.pth")['state_dict']

    model.load_state_dict(weights_tiny)

model.cuda()
model = MMDataParallel(model, device_ids=[0])



#sample = next(iter(data_loader))


# model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))


#sample = next(iter(data_loader))
#sample=None
if not train_setup:
    motion_distribution_targets = {
        # for motion prediction
        "motion_segmentation": sample["motion_segmentation"][0],
        "motion_instance": sample["motion_instance"][0],
        "instance_centerness": sample["instance_centerness"][0],
        "instance_offset": sample["instance_offset"][0],
        "instance_flow": sample["instance_flow"][0],
        "future_egomotion": sample["future_egomotions"][0],
    }

    with torch.no_grad():
        result = model(
            return_loss=False,
            rescale=True,
            img_metas=sample["img_metas"],
            img_inputs=sample["img_inputs"],
            future_egomotions=sample["future_egomotions"],
            motion_targets=motion_distribution_targets,
            img_is_valid=sample["img_is_valid"][0],
        )




cfg.work_dir = "./"
meta = dict()
# log env info
# env_info_dict = collect_env()
# env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
# dash_line = '-' * 60 + '\n'
# logger.info('Environment info:\n' + dash_line + env_info + '\n' +
#             dash_line)
# meta['env_info'] = env_info
# meta['config'] = cfg.pretty_text
# meta['seed'] = 1337
# meta['exp_name'] = "testname"

optimizer = build_optimizer(model, cfg.optimizer)

cfg.runner = {
    'type': 'EpochBasedRunner',
    'max_epochs': 1
}

runner = build_runner(
    cfg.runner,
    default_args=dict(
        model=model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta))

with torch.no_grad():
    runner.run(data_loaders, cfg.workflow)



print("done")
