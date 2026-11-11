"""Numerical checks of the shared adapters against their original upstream paths."""
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from depth_project.io import write_json
from depth_project.models import load_model
from depth_project.models.api import ModelInput
from depth_project.paths import CHECKPOINTS
from depth_project.inference.__main__ import WEIGHTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.manual_seed(20261104)
    rgb=np.random.default_rng(20261104).integers(0,256,(120,160,3),dtype=np.uint8)
    records={}
    for name in ("da2","unidepthv2","vda"):
        adapter=load_model(name,CHECKPOINTS/WEIGHTS[name],input_size=28 if name=="vda" else 518)
        with torch.inference_mode():
            if name=="vda":
                frames=np.random.default_rng(2026).integers(0,256,(45,28,42,3),dtype=np.uint8)
                expected,_=adapter.model.infer_video_depth(frames,1.,input_size=28,fp32=True)
                actual=adapter.forward(ModelInput(rgb=frames)).numpy()
            else:
                expected=adapter.predict(rgb)[None]
                actual=adapter.forward(ModelInput(rgb=[rgb])).numpy()
        np.testing.assert_array_equal(actual,expected)
        records[name]={"shape":list(actual.shape),"max_abs_error":float(np.max(np.abs(actual-expected))),
                       "finite":bool(np.isfinite(actual).all())}
        write_json(args.output,records)
        print(name,records[name],flush=True)
        del actual,expected,adapter
        gc.collect();torch.cuda.empty_cache()


if __name__=="__main__":
    main()
