"""Write the local capability manifest independently of model allocation."""

import argparse
import json
from pathlib import Path
import platform

import torch

from greenspace.cuda.driver import driver


def manifest(device_id: int = 0) -> dict:
    info = driver.get_device_info(device_id)
    info.update(platform=platform.platform(), torch_version=torch.__version__,
                torch_cuda_version=torch.version.cuda,
                torch_green_context_available=hasattr(torch._C, "_CUDAGreenContext"))
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoint_1_hardware.json"))
    args = parser.parse_args()
    data = json.dumps(manifest(args.device), indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data + "\n", encoding="utf-8")
    print(data)
