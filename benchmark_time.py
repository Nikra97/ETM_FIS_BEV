# benchmark_pytorch.py from https://leimao.github.io/blog/PyTorch-Benchmark/
import os
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from custome_logger import setup_custom_logger

logger = setup_custom_logger()
logger.debug("test")

from mmcv import Config

from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


@torch.no_grad()
def measure_time_host(
    model: nn.Module,
    sample,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = True,
) -> float:

    for _ in range(num_warmups):
        _ = model.forward(sample)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start = timer()
        for _ in range(num_repeats):
            _ = model.forward(sample)
        if synchronize:
            torch.cuda.synchronize()
        end = timer()
        elapsed_time_ms = (end - start) * 1000

    else:
        for _ in range(num_repeats):
            start = timer()
            _ = model.forward(sample)
            if synchronize:
                torch.cuda.synchronize()
            end = timer()
            elapsed_time_ms += (end - start) * 1000

    return elapsed_time_ms / num_repeats


@torch.no_grad()
def measure_time_device(
    model: nn.Module,
    sample,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = True,
) -> float:

    for _ in range(num_warmups):
        _ = model.forward(sample)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            _ = model.forward(sample)
        end_event.record()
        if synchronize:
            # This has to be synchronized to compute the elapsed time.
            # Otherwise, there will be runtime error.
            torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

    else:
        for _ in range(num_repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model.forward(sample)
            end_event.record()
            if synchronize:
                # This has to be synchronized to compute the elapsed time.
                # Otherwise, there will be runtime error.
                torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

    return elapsed_time_ms / num_repeats


@torch.no_grad()
def run_inference(model, sample):
    motion_distribution_targets = {
        # for motion prediction
        "motion_segmentation": sample["motion_segmentation"][0],
        "motion_instance": sample["motion_instance"][0],
        "instance_centerness": sample["instance_centerness"][0],
        "instance_offset": sample["instance_offset"][0],
        "instance_flow": sample["instance_flow"][0],
        "future_egomotion": sample["future_egomotions"][0],
    }
    return model(
        return_loss=False,
        rescale=True,
        img_metas=sample["img_metas"],
        img_inputs=sample["img_inputs"],
        future_egomotions=sample["future_egomotions"],
        motion_targets=motion_distribution_targets,
        img_is_valid=sample["img_is_valid"][0],
    )


def main() -> None:

    num_warmups = 5
    num_repeats = 3
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
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    # model = nn.Conv2d(in_channels=input_shape[1], out_channels=256, kernel_size=(5, 5))

    # Input tensor
    # sample = torch.rand(input_shape, device=device)

    iter_dataloader = iter(data_loader)

    torch.cuda.synchronize()

    print("Latency Measurement Using CPU Timer...")
    for continuous_measure in [True]:
        for synchronize in [True]:
            try:
                sample = next(iter_dataloader)
                latency_ms = measure_time_host(
                    model=model,
                    sample=sample,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                    synchronize=synchronize,
                    continuous_measure=continuous_measure,
                )
                print(
                    f"|"
                    f"Synchronization: {synchronize!s:5}| "
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: {latency_ms:.5f} ms| "
                )
            except Exception as e:
                print(
                    f"|"
                    f"Synchronization: {synchronize!s:5}| "
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: N/A     ms| "
                )
            torch.cuda.synchronize()

    print("Latency Measurement Using CUDA Timer...")
    for continuous_measure in [True, False]:
        for synchronize in [True, False]:
            try:
                latency_ms = measure_time_device(
                    model=model,
                    sample=sample,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                    synchronize=synchronize,
                    continuous_measure=continuous_measure,
                )
                print(
                    f"|"
                    f"Synchronization: {synchronize!s:5}| "
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: {latency_ms:.5f} ms| "
                )
            except Exception as e:
                print(
                    f"|"
                    f"Synchronization: {synchronize!s:5}| "
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: N/A     ms| "
                )
            torch.cuda.synchronize()

    print("Latency Measurement Using PyTorch Benchmark...")
    num_threads = 1
    timer = benchmark.Timer(
        stmt="run_inference(model, sample)",
        setup="from __main__ import run_inference",
        globals={"model": model, "sample": sample},
        num_threads=num_threads,
        label="Latency Measurement",
        sub_label="torch.utils.benchmark.",
    )

    profile_result = timer.timeit(num_repeats)
    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    print(f"Latency: {profile_result.mean * 1000:.5f} ms")


if __name__ == "__main__":

    main()
