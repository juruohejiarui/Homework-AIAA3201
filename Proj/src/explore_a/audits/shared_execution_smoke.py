"""Real weights: repeated local/DGX heads, scoring and model-specific RPC checks."""
import argparse
from pathlib import Path

import numpy as np
import torch

from depth_project.io import write_json
from depth_project.models.api import ModelInput
from depth_project.paths import PROJECT, CHECKPOINTS
from depth_project.inference.__main__ import WEIGHTS
from explore_a.config import load_config
from explore_a.data.prepare import load_manifest
from explore_a.online.loading import read_pair, OnlinePairs
from explore_a.online.factory import make_runtime
from explore_a.evaluation.session import entries, bind, forward_frame, reference, head
from explore_a.evaluation.transport import DirectPipeline, RemotePair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=load_config()
    models=[e for e in entries(PROJECT/"maintenance/evaluate_models.json") if e.get("kind","fusion")=="fusion"]
    models.append({"name":"unidepthv2","kind":"frozen","adapter":"unidepthv2",
                   "weights":str(CHECKPOINTS/WEIGHTS["unidepthv2"]),"load_param":{"fp32":True},
                   "specific_param":{"resolution_level":0}})
    bindings=[bind(e,args.output/"snapshots") for e in models]
    runtime=DirectPipeline(config)
    peer=RemotePair(bindings,config)
    selected={}
    for row in load_manifest(config)["samples"]:
        if row["split"]=="dev":selected.setdefault(row["dataset"],row)
    records={}
    try:
        with torch.inference_mode():
            for dataset,row in sorted(selected.items()):
                raw=read_pair(row,config["data"])
                one=forward_frame(bindings,raw,runtime)
                again=forward_frame(bindings,raw,runtime)
                names=[b["entry"]["name"] for b in bindings if "task" in b]
                other=peer.submit(raw,False,True,names).result()
                repeated=peer.submit(raw,False,True,names).result()
                for key in one:
                    np.testing.assert_array_equal(one[key],again[key])
                    np.testing.assert_array_equal(other[key],repeated[key])
                records[dataset]={"id":row["id"],"host_repeat_exact":True,"dgx_repeat_exact":True,
                    "cross_gpu_max_abs":{key:float(np.max(np.abs(one[key]-other[key]))) for key in one}}
                write_json(args.output/"results.json",records)
                print(dataset,records[dataset],flush=True)
            rgb=np.random.default_rng(20261104).integers(0,256,(120,160,3),dtype=np.uint8)
            request=ModelInput(rgb=[rgb],frame_ids=("synthetic",),specific_in={"camera":np.eye(3,dtype=np.float32)[None]})
            b=bindings[-1]
            local=reference(b).forward(request,**b["entry"]["specific_param"])
            result=peer.reference(b,request)
            assert result.frame_ids==request.frame_ids
            assert set(result.specific_out["frames"][0])==set(local.specific_out["frames"][0])
            assert result.prediction.shape==local.prediction.shape
            assert torch.isfinite(result.prediction).all()
            records["reference_rpc"]={"shape":list(result.prediction.shape),
                "specific_out_keys":list(result.specific_out["frames"][0]),
                "cross_gpu_max_abs":float(np.abs(result.numpy()-local.numpy()).max())}
            write_json(args.output/"results.json",records)
        # Exercise a live training model through the same case transport. No optimizer/update.
        from explore_a.evaluation.live import collect
        examples={}
        for row in load_manifest(config)["samples"]:
            if row["split"]=="dev":
                examples.setdefault((row["dataset"],row["sequence"],row["camera"]),row)
            if len(examples)>=8:break
        b=bindings[0]
        from explore_a.training.prediction import make_fusion
        model=make_fusion(b["task"]["fusion"],config).cuda().train()
        model.load_state_dict(b["state"]["model"])
        cpu_rng=torch.get_rng_state().clone()
        outer=make_runtime(config,[b["task"]["flow"]],False)
        try:
            dataset=OnlinePairs(config,list(examples.values()),outer,b["task"]["flow"],phase="dev")
            result,_=collect(model,dataset,config,"cuda","rule",step=b["state"]["step"])
            assert model.training and all(p.requires_grad for p in model.parameters())
            assert torch.equal(cpu_rng,torch.get_rng_state())
            assert result["execution"]["remote"]["remote_pairs"]>0
            records["live_dev"]={"frames":result["counts"]["total_frames"],"selection_score":result["selection_score"],
                                 "mode_grad_rng_preserved":True,"execution":result["execution"]}
            write_json(args.output/"results.json",records)
        finally:
            outer.close()
    finally:
        runtime.close();peer.close()


if __name__=="__main__":
    main()
