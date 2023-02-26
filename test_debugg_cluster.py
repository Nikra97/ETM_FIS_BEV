from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from collections import Counter, OrderedDict
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
from mmcv.runner import wrap_fp16_model
from mmcv import Config
from timeit import default_timer as timer
import numpy as np
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

import tqdm

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(False)


def _parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary infomation.

    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
            which may be a weighted sum of all losses, log_vars contains \
            all the variables to be sent to the logger.
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def import_modules_load_config(cfg_file="beverse_tiny.py", samples_per_gpu=1):
    cfg_path = r"/home/kraussn/EMT_BEV/projects/configs"
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
cfg = import_modules_load_config(cfg_file="motion_detr_tiny_cluster.py")

cfg.data.test["data_root"] = '/home/kraussn/EMT_BEV/data/nuscenes'
dataset = build_dataset(cfg.data.test)


# 3 5 time: 0.746, data_time: 0.042
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=3,
    workers_per_gpu=5,
    dist=False,
    shuffle=False,
)


model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
model.init_weights()


cfg.checkpoint_config.meta = dict(
    mmdet_version=mmdet_version,
    mmseg_version=mmseg_version,
    mmdet3d_version=mmdet3d_version,
    config=cfg.pretty_text,
    CLASSES=dataset.CLASSES,
    PALETTE=dataset.PALETTE  # for segmentors
    if hasattr(dataset, 'PALETTE') else None)


# weights_tiny = torch.load( #
#     "/home/niklas/ETM_BEV/BEVerse/weights/beverse_tiny.pth")["state_dict"]
# model.load_state_dict(weights_tiny)
checkpoint_path = os.path.join(
    "/home/kraussn/EMT_BEV/logs/logs_segmentation/epoch_10.pth")
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES
# palette for visualization in segmentation tasks
if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
elif hasattr(dataset, 'PALETTE'):
    # segmentation dataset has `PALETTE` attribute
    model.PALETTE = dataset.PALETTE


model.cuda()
model = MMDataParallel(model, device_ids=[0])


mmcv.mkdir_or_exist(osp.abspath(
    r"/home/kraussn/EMT_BEV/logs/local_train_debug"))

# init the logger before other steps
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(
    r"/home/kraussn/EMT_BEV/logs/local_train_debug", f'{timestamp}.log')

logger_name = 'mmdet'
logger = get_root_logger(
    log_file=log_file, log_level=cfg.log_level, name=logger_name)


cfg.work_dir = f"/home/kraussn/EMT_BEV/logs/logs_segmentation"
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


logging_interval = 50
model.eval()
dataset = data_loader.dataset

prog_bar = mmcv.ProgressBar(len(dataset))

final_losses = {
    "loss_overall": [],
    "loss_ce": [],
    "loss_mask": [],
    "liss_dice": [],
}

for i, data in enumerate(data_loader):
    with torch.no_grad():
        motion_distribution_targets = {
            # for motion prediction
            'motion_segmentation': data['motion_segmentation'][0],
            'motion_instance': data['motion_instance'][0],
            'instance_centerness': data['instance_centerness'][0],
            'instance_offset': data['instance_offset'][0],
            'instance_flow': data['instance_flow'][0],
            'future_egomotion': data['future_egomotions'][0],
        }
        losses = model(
            return_loss=False,
            rescale=True,
            img_metas=data['img_metas'],
            img_inputs=data['img_inputs'],
            future_egomotions=data['future_egomotions'],
            motion_targets=motion_distribution_targets,
            img_is_valid=data['img_is_valid'][0],
        )

        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        final_losses["loss_overall"].append(loss)

        final_losses["loss_ce"].append(log_vars["loss_ce"])
        final_losses["loss_mask"].append(log_vars["loss_mask"])
        final_losses["liss_dice"].append(log_vars["loss_dice"])

        for _ in range(data_loader.batch_size):
            prog_bar.update()

print("\n")
for loss_name, loss_value in final_losses.items():
    print(f"Test-{loss_name}: {sum(loss_value)/len(loss_value)}")
