"""Gradient equivalence, shared work, stable ownership and independent capacity models."""
from collections import Counter
import copy
import io

import numpy as np
import pytest
import torch

from depth_project.models.api import ModelInput, fusion_forward
from depth_project.models.serialization import pack, unpack
from explore_a.audits.execution_balance import solve
from explore_a.evaluation.session import forward_frame, CaseScores
from explore_a.models.inputs import FusionInputs
from explore_a.online.factory import make_runtime
from explore_a.online.loading import read_pair
from explore_a.online.runtime import FrozenPipeline
from explore_a.training.prediction import make_fusion, fusion_geometry, predict
from test_explore_a_pipeline import online_experiment, fake_backbones


def test_forward_preserves_training_graph_and_matches_direct_model(tmp_path):
    config = online_experiment(tmp_path)
    from explore_a.data.prepare import load_manifest
    row = load_manifest(config)["samples"][0]
    runtime = make_runtime(config, ["farneback"], device="cpu")
    try:
        raw = read_pair(row, config["data"])
        data = {k:v.unsqueeze(0) for k,v in runtime.compute(raw,"farneback").items()}
        a = make_fusion("unet_guided",config)
        b = copy.deepcopy(a)
        geometry = fusion_geometry(data,config)
        geometry["historical"] = data["current_depth"] + torch.linspace(0.,1.,24)[None,None,None,:]
        geometry["rule"] = torch.ones_like(geometry["rule"])
        geometry["hard"] = torch.ones_like(geometry["hard"])
        inputs = FusionInputs.from_pair(data,geometry,torch.ones_like(data["current_depth"]))
        expected = a(inputs)["depth"]
        actual,_ = predict(b,data,config,"rule",geometry)
        torch.testing.assert_close(actual["depth"],expected,rtol=0,atol=0)
        expected.sum().backward();actual["depth"].sum().backward()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        assert actual["depth"].grad_fn is not None
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in b.parameters())
    finally:
        runtime.close()


def test_two_heads_share_frozen_predictions_and_geometry(tmp_path,monkeypatch):
    config=online_experiment(tmp_path)
    from explore_a.data.prepare import load_manifest
    raw=read_pair(load_manifest(config)["samples"][0],config["data"])
    bindings=[{"entry":{"name":str(i)},"config":config,"device":"cpu",
               "task":{"flow":"farneback","occlusion":"rule","fusion":"unet_guided"},
               "model":make_fusion("unet_guided",config)} for i in range(2)]
    calls=[]
    monkeypatch.setattr("explore_a.evaluation.session.fusion_geometry",
                        lambda *a: (calls.append(1),fusion_geometry(*a))[1])
    runtime=make_runtime(config,["farneback"],device="cpu")
    try:
        with torch.inference_mode():
            values=forward_frame(bindings,raw,runtime)
        assert set(values)=={"0","1","raw_canvas"}
        assert len(calls)==1
        assert runtime.statistics()["local_completed"]==3
    finally:
        runtime.close()


def test_routing_uses_input_identity_not_first_occurrence():
    pipeline=object.__new__(FrozenPipeline)
    pipeline.routing={k:{"cache":0.,"remote":.5} for k in ("train","evaluation")}
    pipeline.seed=2026;pipeline.batch=16;pipeline.counts=Counter()
    assert pipeline._origin({},"prediction",False,None,False,1,"dev","a"*64)==pipeline._origin(
        {},"prediction",True,None,False,9999,"dev","a"*64)
    pipeline.evaluation_seed=2026
    first=pipeline._origin({},"prediction",False,None,False,1,"dev","a"*64)
    pipeline.seed=12345
    assert pipeline._origin({},"prediction",False,None,False,1,"dev","a"*64)==first


def test_tree_transport_keeps_specific_tensor_values():
    arrays=[]
    value={"camera":torch.eye(3),"custom":{"mask":np.array([True,False]),"scale":2.}}
    tree=pack(value,arrays)
    from explore_a.remote_online.wire import send,receive
    stream=io.BytesIO();send(stream,{"tree":tree},arrays);stream.seek(0)
    meta,received=receive(stream);restored=unpack(meta["tree"],received)
    np.testing.assert_array_equal(restored["camera"],np.eye(3))
    np.testing.assert_array_equal(restored["custom"]["mask"],[True,False])


def test_tree_transport_preserves_scalars_bfloat16_empty_arrays_and_marker_keys():
    value={"scalar":torch.tensor(2.),"bf16":torch.ones(2,dtype=torch.bfloat16),
           "empty":np.empty((0,3),np.float32),"tuple":(1,2),"marker":{"__array__":4}}
    arrays=[];tree=pack(value,arrays)
    from explore_a.remote_online.wire import send,receive
    stream=io.BytesIO();send(stream,{"tree":tree},arrays);stream.seek(0)
    meta,arrays=receive(stream);restored=unpack(meta["tree"],arrays)
    assert restored["scalar"].shape==torch.Size([])
    assert restored["bf16"].dtype==torch.bfloat16
    assert restored["empty"].shape==(0,3)
    assert restored["tuple"]==(1,2)
    assert restored["marker"]=={"__array__":4}


def test_case_ownership_is_model_independent_and_within_frame_budget():
    from explore_a.evaluation.transport import assign_cases
    config={"data":{"seed":2026},"execution":{"backend":"hybrid"},
            "remote_online":{"enabled":True},"evaluation_execution":{"remote_cases":.2}}
    cases=[{"scope":"dev","dataset":"tum","key":str(i),"frames":[None]*count}
           for i,count in enumerate((10,20,30,40,50,60,70,80))]
    assigned=assign_cases(copy.deepcopy(cases),config)
    reversed_order=assign_cases(copy.deepcopy(list(reversed(cases))),config)
    assert {c["key"]:c["owner"] for c in assigned}=={c["key"]:c["owner"] for c in reversed_order}
    remote=sum(len(c["frames"]) for c in assigned if c["owner"]=="dgx")
    assert 0<remote<=.2*sum(len(c["frames"]) for c in cases)


def test_custom_adapter_accepts_own_constructor_and_is_frozen_in_eval(monkeypatch):
    from depth_project.models.api import ModelOutput
    from explore_a.evaluation.session import reference
    class Custom(torch.nn.Module):
        capability="image"
        def __init__(self,weights):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor(2.))
            self.dropout=torch.nn.Dropout(.5)
        def forward(self,request,**specific_param):
            return ModelOutput(self.dropout(self.weight*torch.ones(1,2,2)),"relative_inverse")
    monkeypatch.setattr("depth_project.models.api.resolve_adapter",lambda name:Custom)
    binding={"entry":{"adapter":"custom:Custom"},"weights":"unused","model":None}
    model=reference(binding)
    assert not model.training and not model.weight.requires_grad
    torch.testing.assert_close(model.forward(ModelInput()).prediction,torch.full((1,2,2),2.))


def test_training_and_evaluation_capacity_are_different():
    result=solve(local_frozen=1.,cached=.3,remote_frozen=1.,head_train=2.,head_eval=.1,remote_full=1.1)
    assert result["train"]["theoretical"]["pair_s"]>=2.
    assert result["evaluation"]["theoretical"]["pair_s"]<1.
    for profile in result.values():
        p=profile["conservative"]
        assert p["remote"] <= profile["theoretical"]["remote"]+1e-9
        assert p["local"]+p["remote"]+p["cache"]==pytest.approx(1.)


def test_case_scorer_scores_identical_baselines_identically():
    truth=np.linspace(1,8,48,dtype=np.float32).reshape(6,8)
    row={"id":"tum:s:rgb:1","dataset":"tum","sequence":"s","camera":"rgb"}
    scorer=CaseScores()
    for name in ("one","two"):
        scorer.add(name,row,.7/truth+.1,truth,np.ones_like(truth,bool),(.1,10.))
    result=scorer.finish()
    assert result["one"]==result["two"]


def test_metric_reference_allows_zero_canvas_padding():
    from explore_a.data.reading import fit_image
    native=np.full((375,1242),3.,np.float32)
    padded,valid=fit_image(native,(256,832))
    assert int((padded==0).sum())==4160
    row={"id":"kitti:sequence:image_02:5","dataset":"kitti","sequence":"sequence","camera":"image_02"}
    scorer=CaseScores()
    scorer.add("unidepthv2",row,padded,np.full_like(padded,3.),valid,(.1,80.),"metric_m")
    measured=scorer.finish()["unidepthv2"]
    assert measured["metric_abs_rel"]==0.
    assert measured["metric_d1"]==1.
    assert measured["absrel_img"]==0.


@pytest.mark.parametrize("invalid",[0.,-1.,float("nan"),float("inf")])
def test_metric_reference_still_rejects_invalid_scored_pixels(invalid):
    prediction=np.full((6,8),3.,np.float32)
    prediction[2,3]=invalid
    row={"id":"test","dataset":"tum","sequence":"sequence","camera":"rgb"}
    scorer=CaseScores()
    try:
        with pytest.raises(ValueError,match="Invalid prediction unidepthv2/test"):
            scorer.add("unidepthv2",row,prediction,np.full_like(prediction,3.),
                       np.ones_like(prediction,bool),(.1,10.),"metric_m")
    finally:
        scorer.close()


def test_session_runs_registry_once_and_resumes_without_inference(tmp_path, fake_backbones):
    import json
    import yaml
    from explore_a.evaluation.session import run
    config=online_experiment(tmp_path)
    config["training"]["source_weights"] = {name: 1 for name in config["data"]["datasets"]}
    cfg=tmp_path/"config.yaml";cfg.write_text(yaml.safe_dump(config))
    model=make_fusion("unet_guided",config)
    checkpoint=tmp_path/"best.pt"
    torch.save({"model":model.state_dict(),"identity":{"training":config["training"],
        "frontend":{"name":"da2","kind":"relative_inverse"},
        "experiment":{"flow":"farneback","occlusion":"rule","fusion":"unet_guided"}},
        "step":1,"best":.1},checkpoint)
    registry=tmp_path/"models.json"
    registry.write_text(json.dumps([{"name":n,"config":str(cfg),"checkpoint":str(checkpoint)}
                                    for n in ("one","two")]))
    result=run(registry,tmp_path/"evaluation",["test"],tmp_path,tmp_path,device="cpu")
    assert {r["model"] for r in result}=={"one","two","raw_canvas"}
    depth,_,_=fake_backbones
    before=depth.calls
    again=run(registry,tmp_path/"evaluation",["test"],tmp_path,tmp_path,device="cpu")
    assert again==result
    assert depth.calls==before
    assert len(list((tmp_path/"evaluation/test").glob("*/*/raw_canvas/metrics.json")))==1
    baseline=next((tmp_path/"evaluation/test").glob("*/*/raw_canvas/metrics.json"))
    stamp=baseline.stat().st_mtime_ns
    registry.write_text(json.dumps([{"name":n,"config":str(cfg),"checkpoint":str(checkpoint)}
                                    for n in ("one","two","three")]))
    expanded=run(registry,tmp_path/"evaluation",["test"],tmp_path,tmp_path,device="cpu")
    assert baseline.stat().st_mtime_ns==stamp
    assert {r["model"] for r in expanded}=={"one","two","three","raw_canvas"}
    assert "absrel_baseline_fixed_video" in next(r for r in expanded if r["model"]=="three")


def test_final_flow_interrupted_legacy_checkpoint_continues_identical_updates(tmp_path, fake_backbones, monkeypatch):
    from pathlib import Path
    from explore_a.training import engine
    from explore_a.observability.observer import RunObserver
    config=online_experiment(tmp_path)
    config["training"].update(steps=4,heldout_every=0)
    complete=copy.deepcopy(config)
    complete["paths"]["runs"]=str(tmp_path/"complete")
    engine.run_training(complete,"farneback","rule","final_flow",device="cpu")
    original_plot=RunObserver.plot
    def interrupt(self, step):
        if step==2:raise KeyboardInterrupt()
        original_plot(self,step)
    monkeypatch.setattr(RunObserver,"plot",interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.run_training(config,"farneback","rule","final_flow",device="cpu")
    root=Path(config["paths"]["runs"])/"farneback-rule-final_flow"
    saved=torch.load(root/"last.pt",weights_only=False)
    assert saved["step"]==2
    saved["identity"].pop("evaluation_protocol")
    torch.save(saved,root/"last.pt")
    old_best=torch.load(root/"best.pt",weights_only=False)
    old_best["identity"].pop("evaluation_protocol")
    old_best["identity"]["dev_selection"]={"protocol":"legacy"}
    torch.save(old_best,root/"best.pt")
    original_validate=engine.validate
    def interrupted_migration(*a,**kw):raise KeyboardInterrupt()
    monkeypatch.setattr(engine,"validate",interrupted_migration)
    with pytest.raises(KeyboardInterrupt):
        engine.run_training(config,"farneback","rule","final_flow",device="cpu")
    assert (root/"best.pt").exists()
    assert torch.load(root/"last.pt",weights_only=False)["step"]==2
    # Also recover an interruption left by the older rename-before-evaluate implementation.
    (root/"best.pt").unlink()
    evaluated_steps=[]
    def observed_validate(*a,**kw):
        evaluated_steps.append(kw.get("step"))
        return original_validate(*a,**kw)
    monkeypatch.setattr(engine,"validate",observed_validate)
    monkeypatch.setattr(RunObserver,"plot",original_plot)
    engine.run_training(config,"farneback","rule","final_flow",device="cpu")
    resumed=torch.load(root/"last.pt",weights_only=False)
    assert evaluated_steps[:2]==[2,old_best["step"]]
    expected=torch.load(tmp_path/"complete/farneback-rule-final_flow/last.pt",weights_only=False)
    for key,value in expected["model"].items():
        torch.testing.assert_close(resumed["model"][key],value,rtol=0,atol=0)
    assert resumed["scheduler"]==expected["scheduler"]
    for key,state in expected["optimizer"]["state"].items():
        for name,value in state.items():
            torch.testing.assert_close(resumed["optimizer"]["state"][key][name],value,rtol=0,atol=0)
