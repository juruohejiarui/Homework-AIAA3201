"""One-checkpoint isolated endpoint measurements with disjoint randomized cache rounds."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
from statistics import median
import time

import torch

from depth_project.io import write_json
from depth_project.paths import PROJECT
from explore_a.config import load_config
from explore_a.data.prepare import load_manifest
from explore_a.online.cache_components import CacheComponents
from explore_a.online.factory import make_runtime
from explore_a.online.loading import OnlinePairs, read_pair
from explore_a.evaluation.session import bind, head, identity, forward_frame
from explore_a.evaluation.transport import RemotePair
from explore_a.training.prediction import predict
from explore_a.training.losses import fusion_loss
from explore_a.audits.execution_balance import solve


def io_bytes():
    return {k: int(v) for k,v in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}


def sample_rounds(config, pairs, rounds, seed, excluded=()):
    from explore_a.training.dev_subset import select_dev
    rows = select_dev([r for r in load_manifest(config)["samples"] if r["split"] == "dev"],
                      config["training"]["dev_sample_fraction"], config["data"]["seed"])
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"],row["sequence"],row["camera"]].append(row)
    chunks = []
    for group in groups.values():
        # Disjoint runs preserve previous-frame reuse without revisiting the same files.
        for offset in range(0,len(group)-16,17):
            chunks.append(group[offset:offset+16])
    random.Random(seed).shuffle(chunks)
    cache = CacheComponents(config)
    chosen, available, scanned = [], 0, 0
    excluded = set(excluded)
    used = {r[k] for r in rows if r["id"] in excluded for k in ("previous_rgb", "current_rgb")}
    needed = pairs*rounds+16
    for chunk in chunks:
        files = {r[k] for r in chunk for k in ("previous_rgb","current_rgb")}
        if files & used:
            continue
        scanned += len(chunk)
        ready = True
        for row in chunk:
            raw = read_pair(row,config["data"])
            if not all([cache.discover(raw,"prediction",old) for old in (True,False)] +
                       [cache.discover(raw,"flow",flow="raft")]):
                ready = False
                break
        if ready:
            available += len(chunk)
            chosen.extend(chunk)
            used.update(files)
        if len(chosen) >= needed:
            break
    if len(chosen) < needed:
        raise ValueError(f"Need {needed} distinct fully cached performance pairs; found {len(chosen)}")
    return chosen[:16], [chosen[16+i*pairs:16+(i+1)*pairs] for i in range(rounds)], available/max(scanned,1)


def synchronize():
    torch.cuda.synchronize()


def measure(config, binding, rows, endpoint, peer, warm=()):
    cfg = copy.deepcopy(config)
    cfg["execution"]["backend"] = "cache" if endpoint == "cache" else "hybrid"
    cfg["execution"]["routing"] = {s:{"cache":float(endpoint=="cache"),
        "remote":float(endpoint=="dgx-frozen")} for s in ("train","evaluation")}
    runtime = make_runtime(cfg,["raft"],False)
    source = OnlinePairs(cfg,[*warm, *rows],runtime,"raft",phase="dev")
    stream = source.raw_stream(submit=endpoint!="dgx-full")
    before = io_bytes(); synchronize(); began = time.perf_counter(); heads=0.; initial={}
    last = None
    try:
        for index,raw in stream:
            if index == len(warm):
                synchronize(); before=io_bytes(); began=time.perf_counter(); heads=0.
                initial=runtime.statistics()
            if endpoint == "dgx-full":
                runtime.cancel_ticket(raw["frozen_ticket"])
                peer.submit(raw,False,True,[binding["entry"]["name"]]).result()
            else:
                base = runtime.depth_pair(raw)
                data = runtime.add_flow(raw,base,"raft")
                data = {k:v.unsqueeze(0) for k,v in data.items()}
                if endpoint != "dgx-frozen":
                    synchronize(); start=time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda",enabled=cfg["training"]["amp"]):
                        predict(head(binding),data,cfg,"rule")
                    synchronize(); heads+=time.perf_counter()-start
                last=data
        synchronize(); wall=time.perf_counter()-began
        stats={k:v-initial.get(k,0) for k,v in runtime.statistics().items()}; after=io_bytes()
        return {"endpoint":endpoint,"pairs":len(rows),"wall_s":wall,"pair_s":wall/len(rows),
                "head_s":heads/len(rows),"disk_read_bytes":after["read_bytes"]-before["read_bytes"],
                "logical_read_bytes":after["rchar"]-before["rchar"],"counts":stats},last
    finally:
        stream.close(); runtime.close()


def training_head(binding,data):
    """Zero-LR scratch optimizer measures backward/update cost without changing trained weights."""
    model = copy.deepcopy(head(binding)).train().requires_grad_(True)
    batch = binding["config"]["training"]["effective_batch"]
    data={k:v.repeat(batch,*([1]*(v.ndim-1))) for k,v in data.items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.)
    times=[]
    try:
        for i in range(6):
            optimizer.zero_grad(set_to_none=True); synchronize(); began=time.perf_counter()
            with torch.autocast("cuda",enabled=binding["config"]["training"]["amp"]):
                output,_=predict(model,data,binding["config"],"rule")
                loss=fusion_loss(output,data,binding["config"]["training"]).total
            loss.backward(); optimizer.step(); synchronize()
            if i>=2:times.append((time.perf_counter()-began)/batch)
    finally:
        del optimizer,model
    return median(times)


def validate_mixed(config, binding, rows, profile, training, peer, warm=()):
    """Measure the selected allocation, including scratch backward for training."""
    from contextlib import closing
    from concurrent.futures import ThreadPoolExecutor
    from explore_a.evaluation.transport import assign_cases
    cfg=copy.deepcopy(config)
    cfg["execution"]["backend"]="hybrid"
    cfg["execution"]["routing"]={"train":{"cache":profile["cache"],"remote":profile["remote"]},
        "evaluation":{"cache":profile["cache"]/max(1-profile["remote"],1e-9),"remote":0.}}
    cfg["evaluation_execution"]={"remote_cases":profile["remote"]}
    runtime=make_runtime(cfg,["raft"],False)
    model=copy.deepcopy(head(binding)).train(training).requires_grad_(training)
    local_binding={**binding,"config":cfg,"model":model}
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.) if training else None
    scaler=torch.amp.GradScaler("cuda",enabled=cfg["training"]["amp"])
    owners=defaultdict(int)
    def train_rows(selected):
        source=OnlinePairs(cfg,selected,runtime,"raft",phase="train")
        accumulation=cfg["training"]["effective_batch"]//cfg["training"]["microbatch"]
        optimizer.zero_grad(set_to_none=True)
        n=0
        with closing(source.iter_batches()) as stream:
            for indices,data in stream:
                with torch.autocast("cuda",enabled=cfg["training"]["amp"]):
                    output,_=predict(model,data,cfg,"rule")
                    loss=fusion_loss(output,data,cfg["training"]).total/accumulation
                scaler.scale(loss).backward();n+=1
                if n%accumulation==0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["gradient_clip"])
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                owners["host_head"]+=len(indices)
    def eval_rows(selected):
        groups=defaultdict(list)
        for row in selected:
            groups[row["dataset"],row["sequence"],row["camera"],row.get("evaluation_clip","full-sequence")].append(row)
        cases=assign_cases([{"scope":"dev","dataset":key[0],"key":identity(key),"rows":rs}
                            for key,rs in groups.items()],cfg)
        host_rows=[r for c in cases if c["owner"]=="host" for r in c["rows"]]
        remote_rows=[r for c in cases if c["owner"]=="dgx" for r in c["rows"]]
        def remote_work():
            for row in remote_rows:
                raw=read_pair(row,cfg["data"])
                peer.submit(raw,False,True,[binding["entry"]["name"]]).result()
            owners["dgx"]+=len(remote_rows)
        with ThreadPoolExecutor(1) as scheduler:
            future=scheduler.submit(remote_work)
            source=OnlinePairs(cfg,host_rows,runtime,"raft",phase="dev")
            with closing(source.raw_stream()) as stream, torch.inference_mode():
                for _,raw in stream:
                    forward_frame([local_binding],raw,runtime)
                    owners["host_head"]+=1
            future.result()
    try:
        execute=train_rows if training else eval_rows
        if warm:execute(warm)
        synchronize(); before=io_bytes(); began=time.perf_counter();initial=runtime.statistics();owners.clear()
        execute(rows)
        synchronize(); wall=time.perf_counter()-began;after=io_bytes()
        return {"phase":"train" if training else "evaluation","pairs":len(rows),"pair_s":wall/len(rows),
                "wall_s":wall,"allocation":profile,"owners":dict(owners),
                "counts":{k:v-initial.get(k,0) for k,v in runtime.statistics().items()},
                "disk_read_bytes":after["read_bytes"]-before["read_bytes"],
                "note":"warm-up excluded; training uses configured microbatch, AMP, backward, clipping and scratch zero-LR updates"}
    finally:
        runtime.close();del optimizer,model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=PROJECT/"configs/explore_a/mixed_h5_da2.yaml")
    p.add_argument("--checkpoint",type=Path,default=PROJECT/"outputs/explore_a/mixed_h5_balanced/runs_da2_guided_30k/raft-rule-unet_guided/best.pt")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pairs",type=int,default=256);p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--seed",type=int,default=20261104)
    p.add_argument("--reuse-costs",type=Path,help="Reuse endpoint measurements and recheck the selected schedule")
    p.add_argument("--exclude-protocol",type=Path,action="append",default=[])
    args=p.parse_args();cfg=load_config(args.config)
    if args.pairs < 16 or args.rounds < 1:
        p.error("pairs >=16 and rounds >=1")
    b=bind({"name":"tuning_raft_guided","checkpoint":str(args.checkpoint),"config":str(args.config)},args.output/"snapshots")
    excluded=[]
    for path in args.exclude_protocol:
        previous=json.loads(path.read_text())
        excluded.extend(previous.get("warmup_ids",[]))
        for key in ("sample_ids","validation_sample_ids"):
            excluded.extend(r for group in previous.get(key,[]) for r in group)
    warm,all_rounds,availability=sample_rounds(cfg,args.pairs,2 if args.reuse_costs else args.rounds+2,args.seed,excluded)
    rounds=[] if args.reuse_costs else all_rounds[:args.rounds]
    write_json(args.output/"protocol.json",{"seed":args.seed,"sample_ids":[[r["id"] for r in rs] for rs in rounds],
               "validation_sample_ids":[[r["id"] for r in rs] for rs in all_rounds[-2:]],"warmup_ids":[r["id"] for r in warm],
               "cache_availability":availability,"checkpoint":str(args.checkpoint),"note":"random disjoint files, not guaranteed cold OS page cache"})
    peer=RemotePair([b],cfg);records=[];last=None
    try:
        from explore_a.remote_online.sync import ensure_synchronized
        ensure_synchronized(cfg)
        for i,rows in enumerate(rounds):
            for endpoint in ("cache","local","dgx-frozen","dgx-full"):
                record,data=measure(cfg,b,rows,endpoint,peer,warm)
                record["round"]=i;records.append(record)
                if data is not None:last=data
                write_json(args.output/"measurements.json",records)
                print(record["round"],endpoint,record["pair_s"],record["disk_read_bytes"],flush=True)
        if args.reuse_costs:
            numbers=json.loads(args.reuse_costs.read_text())["costs"]
        else:
            costs={key:[r for r in records if r["endpoint"]==key] for key in ("cache","local","dgx-frozen","dgx-full")}
            h=median(r["head_s"] for r in costs["local"])
            numbers={"local_frozen":median(r["pair_s"]-r["head_s"] for r in costs["local"]),
                 "cached":max(r["pair_s"]-r["head_s"] for r in costs["cache"]),
                 "remote_frozen":median(r["pair_s"] for r in costs["dgx-frozen"]),
                 "remote_full":median(r["pair_s"] for r in costs["dgx-full"]),
                 "head_eval":h,"head_train":training_head(b,last),"availability":availability}
        result={"costs":numbers, "balance":solve(**numbers)}
        write_json(args.output/"balance.json",result);print(json.dumps(result,indent=2),flush=True)
        checks=[]
        for training,rows in zip((True,False),all_rounds[-2:]):
            profile=result["balance"]["train" if training else "evaluation"]["conservative"]
            for label,allocation in (("conservative",profile),("local_reference",{"local":1.,"cache":0.,"remote":0.})):
                check=validate_mixed(cfg,b,rows,allocation,training,peer,warm)
                check["profile"]=label;checks.append(check)
                write_json(args.output/"mixed_validation.json",checks)
                print(json.dumps(check),flush=True)
    finally:
        peer.close()


if __name__ == "__main__":
    main()
