"""Portable orchestration of the included original training functions.

Evidence is read-only; every new training artifact is written below rerun/.
Run phases as separate Python invocations to preserve thread initialization.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HOME = Path(__file__).resolve().parent
EXP = HOME / "experiments/ctd_sgformer"
SPLITS = EXP / "revision_v2/splits"
RUN = HOME / "rerun"
sys.path[:0] = [str(EXP), str(EXP / "revision_v2")]
METHODS = ("sgc", "fixed_tuned", "fixed_budget")

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def paths():
    out = sorted(SPLITS.glob("*_seed*.npz"))
    if len(out) != 10: raise RuntimeError("Ten frozen splits required")
    return out
def config_runtime(device_mode="auto"):
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cpu" if device_mode=="cpu" else "cuda" if torch.cuda.is_available() else "cpu")
def freeze(path, value):
    if path.exists():
        if read(path) != value: raise RuntimeError("Rerun inputs changed; use a fresh extracted copy")
    else: write(path, value)
def baseline_runtime():
    import run_experiments as r
    from models_v2 import make_model
    r.MASK = {"sgc":"none", "fixed_tuned":"none", "fixed_budget":"budget"}
    r.model_for = make_model
    return r
def baseline_configs(method):
    from run_revision import configs
    return configs(method)
def baseline_fit(device, smoke=False):
    r = baseline_runtime()
    target = RUN / ("smoke/baselines" if smoke else "baselines")
    if (target / "test_results.json").exists(): raise RuntimeError("Test already evaluated; do not retune")
    freeze(target / "portable_plan.json", {"methods":list(METHODS),
        "split_sha256":{p.stem:sha(p) for p in paths()},
        "configs":{m:baseline_configs(m) for m in METHODS}, "smoke":smoke})
    selections = []
    for p in (paths()[:1] if smoke else paths()):
        for m in METHODS:
            records = []
            for i,cfg in enumerate(baseline_configs(m)[:1] if smoke else baseline_configs(m)):
                records.append(r.fit_one(p, m, i, cfg, target / "fits" / p.stem / m / f"config{i}", device,
                                         smoke_epochs=2 if smoke else None))
            best = max(records, key=lambda x:(x["validation"]["macro_ap"], -x["config_index"]))
            selections.append(best)
    write(target / "selections.json", selections)
def baseline_test(device):
    import numpy as np, torch
    from protocol import make_context, disease_metrics
    r = baseline_runtime(); target = RUN / "baselines"
    if (target / "test_results.json").exists(): raise RuntimeError("Locked evaluation already exists")
    selections = read(target / "selections.json")
    if len(selections) != 30: raise RuntimeError("Fit all baseline configurations first")
    records = []
    for choice in selections:
        p = SPLITS / f"{choice['split']}.npz"
        folder = target / "fits" / p.stem / choice["method"] / f"config{choice['config_index']}"
        if sha(folder / "best.pt") != choice["checkpoint_sha256"]: raise RuntimeError("Checkpoint changed")
        with np.load(p, allow_pickle=False) as d:
            fm=r.tensor(d["metabolite_features"],device,torch.float32); fd=r.tensor(d["disease_features"],device,torch.float32)
            pairs=d["test_pairs"].copy(); labels=d["test_labels"].copy()
            context=make_context(d["train_pos"],np.empty((0,2),np.int64),len(fm),len(fd),d["train_diseases"],0)["none"]
        model=r.model_for(choice["method"],choice["configuration"],fm,fd).to(device)
        saved=torch.load(folder/"best.pt",map_location=device,weights_only=True)
        model.load_state_dict(saved["model_state"])
        logits=r.predict(model,fm,fd,r.graph_tensors(context,device),pairs,device)
        result={**choice,"test":disease_metrics(pairs,labels,logits,len(fd))}
        records.append(result)
        np.savez_compressed(folder/"test_predictions.npz",pairs=pairs,labels=labels,logits=logits)
    write(target/"test_results.json",records)
def full_fit(device, smoke=False):
    import run_ego_neighbor_validation as r
    target=RUN/("smoke/full" if smoke else "full")
    if (RUN/"full_test/test_results.json").exists() and not smoke: raise RuntimeError("Test already evaluated; do not retune")
    r.OUT=target; r.HIDDEN=64
    if smoke: r.MAX_EPOCHS=2
    freeze(target/"validation_only_plan.json",{"purpose":"fresh portable rerun; fixed historical benchmark",
        "source_hashes":{name:sha(EXP/name) for name in ("models.py","protocol.py","ego_neighbor_model.py","run_ego_neighbor_validation.py")},
        "split_sha256":{p.stem:sha(p) for p in paths()}, "smoke":smoke})
    selections=[]; all_results=[]
    for p in (paths()[:1] if smoke else paths()):
        rows=[r._fit_one(p,"sgc_mlp",lr,device) for lr in ((.001,) if smoke else (.001,.0003))]
        all_results.extend(rows)
        selections.append(max(rows,key=lambda x:(x["validation"]["macro_ap"],-x["lr"])))
    write(target/"validation_results.json",all_results)
    write(target/"selections.json",selections)
    summary=[]
    for setting in ("random_edge","cold_disease"):
        summary.append({"setting":setting,"method":"SGC + interaction MLP",
            "selected_lr_by_split":{x["split"]:x["lr"] for x in selections if x["split"].startswith(setting)}})
    write(target/"validation_summary.json",summary)
def full_test(device):
    import numpy as np, torch
    import run_ego_neighbor_validation as r
    from ego_neighbor_model import EgoNeighborLinkPredictor
    from protocol import make_context
    target=RUN/"full_test"
    if target.exists(): raise FileExistsError(target)
    selected=read(RUN/"full/selections.json")
    if len(selected)!=10: raise RuntimeError("Fit both full-model rates on all ten splits first")
    target.mkdir(parents=True)
    plan={"purpose":"fresh rerun frozen-checkpoint evaluation", "selected_learning_rate_by_split":{},
          "split_sha256":{},"checkpoint_sha256":{}}
    records=[]
    for choice in selected:
        p=SPLITS/f"{choice['split']}.npz"
        checkpoint=RUN/"full/fits"/p.stem/"sgc_mlp"/f"lr{choice['lr']:g}"/"best.pt"
        if sha(checkpoint)!=choice["checkpoint_sha256"]: raise RuntimeError("Checkpoint changed")
        plan["selected_learning_rate_by_split"][p.stem]=choice["lr"]
        plan["split_sha256"][p.stem]=sha(p); plan["checkpoint_sha256"][p.stem]=sha(checkpoint)
    write(target/"test_evaluation_plan.json",plan)
    for choice in selected:
        p=SPLITS/f"{choice['split']}.npz"
        with np.load(p,allow_pickle=False) as d:
            fm=torch.as_tensor(d["metabolite_features"],dtype=torch.float32,device=device)
            fd=torch.as_tensor(d["disease_features"],dtype=torch.float32,device=device)
            pairs=d["test_pairs"].copy();labels=d["test_labels"].copy()
            context=make_context(d["train_pos"],np.empty((0,2),np.int64),len(fm),len(fd),d["train_diseases"],0)["none"]
        model=EgoNeighborLinkPredictor(fm.shape[1],fd.shape[1],len(fm),len(fd),hidden=64,dropout=.2,variant="sgc_mlp").to(device)
        checkpoint=RUN/"full/fits"/p.stem/"sgc_mlp"/f"lr{choice['lr']:g}"/"best.pt"
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)["state_dict"])
        logits,metrics=r._validation(model,fm,fd,r._graph(context,device),pairs,labels,len(fd),device)
        records.append({"split":p.stem,"learning_rate_selected_on_validation":choice["lr"],
            "best_epoch_selected_on_validation":choice["best_epoch"],"parameters":choice["parameters"],"test":metrics})
        np.savez_compressed(target/f"{p.stem}_predictions.npz",pairs=pairs,labels=labels,logits=logits)
    write(target/"test_results.json",records)
def feature_phase(phase,device_mode="auto"):
    source=EXP/"reviewer_revision_20261023/run_decoder_ablation.py"
    spec=importlib.util.spec_from_file_location("portable_review_ablation",source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=EXP;module.HERE=RUN/"reviewer_ablation"
    module.HERE.mkdir(parents=True,exist_ok=True)
    module.SUPPORT=module.HERE/"frozen_support";module.SPLITS=SPLITS
    module.OLD_VAL=RUN/"full";module.OLD_TEST=RUN/"full_test"
    if phase=="ablation-prepare": module.prepare()
    else:
        runtime=module.load_runtime()
        if device_mode=="cpu":
            runtime=(*runtime[:-1],runtime[1].device("cpu"))
        (module.train if phase=="ablation-train" else module.test)(*runtime)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",required=True,choices=("smoke","baselines-fit","baselines-test","full-fit","full-test","ablation-prepare","ablation-train","ablation-test"))
    p.add_argument("--device",choices=("auto","cpu"),default="auto")
    a=p.parse_args()
    if a.phase.startswith("ablation-"): feature_phase(a.phase,a.device); return
    device=config_runtime(a.device)
    if a.phase=="smoke": baseline_fit(device,True);full_fit(device,True)
    elif a.phase=="baselines-fit": baseline_fit(device)
    elif a.phase=="baselines-test": baseline_test(device)
    elif a.phase=="full-fit": full_fit(device)
    elif a.phase=="full-test": full_test(device)
if __name__=="__main__": main()
