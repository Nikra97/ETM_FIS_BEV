# inspired by https://github.com/pytorch/kineto/blob/main/tb_plugin/examples/resnet50_profiler_api.py
import os

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from custome_logger import setup_custom_logger
import logging

logger = setup_custom_logger()
logger.debug("Profiler")


from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


def main() -> None:

    num_warmups = 100
    num_repeats = 1000
    # Change to C x 1 x 3 x 1600 x 900
    # 704×256 Tiny
    # 1408×512
    # input_shape = (6, 1600, 900, 3)

    device = torch.device("cuda:0")
    cfg = Config.fromfile(
        r"/home/niklas/ETM_BEV/BEVerse/projects/configs/beverse_tiny.py"
    )

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

    torch.backends.cudnn.benchmark = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    wrap_fp16_model(model)
    model = fuse_conv_bn(model)
    model.cuda(device)
    model.cuda()
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./logs_profiler", worker_name="worker0"
        ),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True,
    ) as p:
        for i, (data) in enumerate(data_loader):
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)

            if i + 1 >= 4:
                break
            p.step()


if __name__ == "__main__":

    main()
