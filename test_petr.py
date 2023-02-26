import sys
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv import Config, DictAction
import warnings
from timeit import default_timer as timer
import torch.utils.benchmark as benchmark
import torch.nn as nn
import torch
import os
import time
import logging
from custome_logger import setup_custom_logger
logger = setup_custom_logger()
logger.debug("test")
print(sys.path)

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


torch.backends.cudnn.benchmark = True


cfg = import_modules_load_config(
    cfg_file=r"petr/petr_r50dcn_gridmask_p4.py")  # petrv2/petrv2_vovnet_gridmask_p4_800x320.py

cfg.data_root = '/home/niklas/ETM_BEV/BEVerse/data/nuscenes/'
cfg["data"]["test"]["data_root"] = '/home/niklas/ETM_BEV/BEVerse/data/nuscenes/'
cfg["data"]["test"]["ann_file"] = '/home/niklas/ETM_BEV/BEVerse/data/nuscenes/nuscenes_infos_val.pkl'


dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)

sample = next(iter(data_loader))


model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))


model.cuda()
model = MMDataParallel(model, device_ids=[0])


with torch.no_grad():
    result = model(
        return_loss=False,
        rescale=True,
        img_metas=sample['img_metas'],
        img=sample['img'],
        #future_egomotions=sample['future_egomotions'],
        #motion_targets=motion_distribution_targets,
        #img_is_valid=sample['img_is_valid'][0],
    )
