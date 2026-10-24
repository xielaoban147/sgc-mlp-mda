"""Frozen exploratory revision. Fit and evaluate are separate commands."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import run_experiments as runner
from protocol import _load_canonical_dataset, _split_arrays
from models_v2 import make_model

SEEDS = (17, 42, 2026, 31415, 27182)
METHODS = ['mlp','sgc','gatv2','fixed_none','fixed_budget','fixed_uniform',
           'hard_none','hard_budget','hard_uniform','soft_budget',
           'residual_budget','residual_none','residual_fixed_budget','global_only','fixed_tuned']
MASK = {m: 'budget' if m.endswith('_budget') else 'uniform' if m.endswith('_uniform') else 'none' for m in METHODS}
RUN = BASE / 'runs/20261016_revision_v2'

def configs(method):
    values = []
    for alpha in ([0.2, 0.5, 0.8] if method == 'fixed_tuned' else [0.5]):
        for lr in (0.001, 0.0003):
            cfg = runner.settings('mlp')[0]
            cfg.update(alpha=alpha, lr=lr)
            values.append(cfg)
    return values

def setup():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.METHODS = METHODS
    runner.MASK = MASK
    runner.model_for = make_model

def prepare():
    split_dir = HERE / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_canonical_dataset(BASE / 'data/features/dataset.npz')
    for mode in ('random_edge','cold_disease'):
        for seed in SEEDS:
            target = split_dir / f'{mode}_seed{seed}.npz'
            if target.exists():
                continue
            if seed in (17,42,2026):
                shutil.copy2(BASE / 'data/splits' / target.name, target)
            else:
                np.savez_compressed(target, **_split_arrays(dataset, mode, seed))
    return sorted(split_dir.glob('*.npz'))

def freeze(device):
    splits = prepare()
    RUN.mkdir(parents=True, exist_ok=True)
    source_names = ['models.py','protocol.py','run_experiments.py',
                    'revision_v2/models_v2.py','revision_v2/run_revision.py']
    plan = dict(created_utc=runner.now(), protocol_version='revision-v2-exploratory',
                dataset=str((BASE/'data/features/dataset.npz').resolve()),
                dataset_sha256=runner.sha(BASE/'data/features/dataset.npz'),
                source_sha256={n:runner.sha(BASE/n) for n in source_names},
                splits=[dict(path=str(p.resolve()),sha256=runner.sha(p)) for p in splits],
                methods=METHODS, configurations={m:configs(m) for m in METHODS},
                environment=runner.environment(device), seeds=SEEDS,
                status='Review-driven exploratory extension: previous test results informed design; additional seeds are not external confirmation.',
                selection='validation macro AP only; max200/patience20; all configurations fit before opening revision test scores',
                primary_candidate='residual_budget (predeclared; not replaceable by test winner)',
                comparisons=['hard_none vs fixed_none','hard_budget vs hard_none','hard_budget vs hard_uniform',
                             'soft_budget vs hard_budget','residual_budget vs hard_budget',
                             'residual_budget vs residual_none','residual_budget vs residual_fixed_budget',
                             'residual_budget vs sgc','residual_budget vs fixed_tuned'],
                inference='Report all five paired differences, mean/SD and win counts; no significance or external-generalization claim.',
                tuning='Shared two learning rates; mechanistic fixed alpha0.5. fixed_tuned baseline gets alpha0.2/0.5/0.8 x same two rates.',
                secondary_baseline='One compatible domain adaptation, audited and frozen separately before its test evaluation.',
                hardware_budget='single GPU process, four CPU threads; small shallow models; no parallel GPU jobs')
    path = RUN/'frozen_plan.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        for key in ('source_sha256','dataset_sha256','splits','methods','configurations'):
            if old[key] != plan[key]:
                raise RuntimeError(f'Frozen plan changed: {key}')
        return old
    runner.write_json(path,plan)
    return plan

def report(records):
    rows=[]
    for setting in ('random_edge','cold_disease'):
        for method in METHODS:
            rr=[r for r in records if r['setting']==setting and r['method']==method]
            assert len(rr)==len(SEEDS)
            row=dict(setting=setting,method=method,n=len(rr))
            for key in ('macro_ap','macro_recall20','macro_auroc'):
                v=np.array([r['test'][key] for r in rr])
                row.update({key+'_mean':float(v.mean()),key+'_std':float(v.std(ddof=1))})
            for key in ('parameters','train_total_seconds','peak_cuda_allocated_mib','tuning_total_seconds'):
                row[key+'_mean']=float(np.mean([r[key] for r in rr]))
            rows.append(row)
    runner.write_json(RUN/'summary.json',rows)
    pairs=[]
    for setting in ('random_edge','cold_disease'):
        for a,b in [('hard_none','fixed_none'),('hard_budget','hard_none'),('hard_budget','hard_uniform'),
                    ('soft_budget','hard_budget'),('residual_budget','hard_budget'),
                    ('residual_budget','residual_none'),('residual_budget','residual_fixed_budget'),
                    ('residual_budget','sgc'),('residual_budget','fixed_tuned')]:
            delta=[]
            for seed in SEEDS:
                get=lambda m:next(r['test']['macro_ap'] for r in records if r['setting']==setting and r['method']==m and r['seed']==seed)
                delta.append(get(a)-get(b))
            pairs.append(dict(setting=setting,a=a,b=b,seeds=SEEDS,ap_deltas=delta,
                              mean=float(np.mean(delta)),sd=float(np.std(delta,ddof=1)),wins=int((np.array(delta)>0).sum())))
    runner.write_json(RUN/'paired_comparisons.json',pairs)

def fit(plan, device):
    if (RUN/'test_evaluation_started.json').exists():
        raise RuntimeError('Revision test evaluation already started; tuning is locked.')
    runner.verify_plan(plan)
    count=0
    for item in plan['splits']:
        path=Path(item['path'])
        for method in plan['methods']:
            results=[]
            for index,cfg in enumerate(plan['configurations'][method]):
                target=RUN/'fits'/path.stem/method/f'config{index}'
                results.append(runner.fit_one(path,method,index,cfg,target,device))
                count+=1
            index=max(range(len(results)),key=lambda i:(results[i]['validation']['macro_ap'],-i))
            runner.write_json(RUN/'selections'/path.stem/f'{method}.json',
                              dict(split=path.stem,method=method,config_index=index,
                                   selected_validation_ap=results[index]['validation']['macro_ap'],
                                   tuning_total_seconds=sum(r['train_total_seconds'] for r in results),
                                   configuration_count=len(results)))
    runner.write_json(RUN/'training_complete.json',dict(completed_utc=runner.now(),configurations=count,test_arrays_opened=False))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['freeze','smoke','fit','evaluate','report'])
    a=p.parse_args()
    setup()
    device=torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if a.stage=='smoke':
        paths=prepare()
        path=next(p for p in paths if p.name=='cold_disease_seed17.npz')
        for method in METHODS:
            runner.fit_one(path,method,0,configs(method)[0],RUN/'smoke'/method,device,smoke_epochs=2)
        return
    plan=freeze(device)
    if a.stage=='fit': fit(plan,device)
    if a.stage=='evaluate': report(runner.evaluate(plan,RUN,device))
    if a.stage=='report': report(json.loads((RUN/'all_results.json').read_text(encoding='utf-8')))
    print(json.dumps(dict(event='revision_stage_complete',stage=a.stage)),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
