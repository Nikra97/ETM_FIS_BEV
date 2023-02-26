# Copyright (c) OpenMMLab. All rights reserved.
from black import out
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet3d.apis import single_gpu_test
from projects.mmdet3d_plugin.tools import single_gpu_test as mtl_single_gpu_test
from projects.mmdet3d_plugin.tools import multi_gpu_test as mtl_multi_gpu_test

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor


cfg = Config.fromfile(r"/home/niklas/ETM_BEV/BEVerse/projects/configs/beverse_tiny.py")
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
            _module_dir = os.path.dirname(
                r"/home/niklas/ETM_BEV/BEVerse/projects/configs/beverse_tiny.py"
            )
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

dataset = build_dataset(cfg.data.train)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,
)
print(type(dataset))
# print(type(data_loader))
# testsample = next(iter(data_loader))
print("Done")
