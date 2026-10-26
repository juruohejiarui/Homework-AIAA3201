"""Small ARM/GPU/attention compatibility report; never constructs project models."""
import argparse
import importlib.metadata as metadata
import platform

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from depth_project.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert torch.__version__ == "2.12.0+cu130", torch.__version__
    import torchvision
    assert torchvision.__version__ == "0.27.0+cu130", torchvision.__version__
    report = {"python": platform.python_version(), "architecture": platform.machine(),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda_available": torch.cuda.is_available(), "packages": {name: metadata.version(name)
                  for name in ("numpy", "scipy", "opencv-python", "timm", "flash-attn")}}
    if not report["cuda_available"]:
        raise RuntimeError("ARM torch cannot access GB10")
    q = torch.ones(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    try:
        from flash_attn import flash_attn_func
        result = flash_attn_func(q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2))
        report["flash_attn_2"] = bool(torch.isfinite(result).all())
    except (ImportError, OSError, RuntimeError) as error:
        report["flash_attn_2"] = str(error)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        report["torch_sdpa_flash"] = bool(torch.isfinite(result).all())
    except RuntimeError as error:
        report["torch_sdpa_flash"] = str(error)
    report["installed"] = {item.metadata["Name"].lower(): item.version
                           for item in metadata.distributions() if item.metadata.get("Name")}
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
