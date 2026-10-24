"""Frozen validation selection followed by separate final-test evaluation.

Run with the project's virtual environment. No test arrays are loaded by fit().
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from models import create_model
from protocol import make_context, disease_metrics, validation_macro_ap

HERE = Path(__file__).resolve().parent
METHODS = ['mlp', 'sgc', 'gatv2', 'sgformer', 'full', 'fixed_budget',
           'gate_uniform', 'fixed_uniform', 'degree_budget', 'global_only']
MAIN = ['mlp', 'sgc', 'gatv2', 'sgformer', 'full']
ABLATION = ['full', 'fixed_budget', 'gate_uniform', 'fixed_uniform', 'degree_budget', 'global_only']
MASK = {m: ('budget' if m in ('full', 'fixed_budget', 'degree_budget')
            else 'uniform' if m in ('gate_uniform', 'fixed_uniform') else 'none') for m in METHODS}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

def settings(method):
    base = dict(hidden=64, layers=1, dropout=0.2, alpha=0.5, c=1.0,
                lr=0.001, max_epochs=200, patience=20, batch_positive=128,
                optimizer='Adam', weight_decay=0.0, output_dim=64)
    variants = []
    for index in range(2):
        cfg = dict(base)
        if method in ('sgformer', 'fixed_budget', 'fixed_uniform'):
            cfg['alpha'] = [0.2, 0.8][index]
        elif method == 'degree_budget':
            cfg['c'] = [1.0, 3.0][index]
        else:
            cfg['lr'] = [0.001, 0.0003][index]
        variants.append(cfg)
    return variants

def environment(device):
    import rdkit, sklearn, torch_geometric
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                pyg=torch_geometric.__version__, rdkit=rdkit.__version__, sklearn=sklearn.__version__,
                numpy=np.__version__, device=str(device), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                gpu_total_mib=torch.cuda.get_device_properties(0).total_memory / 2**20 if device.type == 'cuda' else None,
                torch_threads=torch.get_num_threads(), deterministic_algorithms=True,
                cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'], tf32=False)

def freeze(run_dir, dataset, splits, device):
    design = HERE.parents[1] / 'outputs/代谢物疾病研究入门/五篇论文汇总与研究Idea.md'
    design_snapshot = run_dir / 'research_design.md'
    if not design_snapshot.exists():
        shutil.copy2(design, design_snapshot)
    feature_manifest = dataset.parent / 'manifest.json'
    plan = dict(client_date='2026-10-14', created_utc=now(),
                task='CTD metabolite-disease link prediction', protocol_version=1,
                dataset=str(dataset.resolve()), dataset_sha256=sha(dataset),
                research_design_sha256=sha(design_snapshot), feature_manifest_sha256=sha(feature_manifest),
                source_sha256={name: sha(HERE / name) for name in ('features.py', 'protocol.py', 'models.py', 'run_experiments.py')},
                splits=[dict(path=str(p.resolve()), sha256=sha(p)) for p in splits],
                methods=METHODS, configurations={m: settings(m) for m in METHODS},
                model_seeds='Each split seed reused for initialization; fresh model per config',
                selection='Best validation macro AP checkpoint; best configuration per method and split; tie lower configuration index',
                mask='Remove query then disease budget or matched uniform; independent shared SeedSequence per epoch/batch',
                fit_never_loads_test_arrays=True, train_plus_validation_retraining=False,
                formal_test_policy='All configuration training completes before formal model test scoring; independent split-label auditing is allowed and does not use model scores',
                environment=environment(device))
    target = run_dir / 'frozen_plan.json'
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        for key in ('dataset_sha256', 'feature_manifest_sha256', 'research_design_sha256', 'source_sha256', 'splits', 'methods', 'configurations'):
            if old[key] != plan[key]:
                raise RuntimeError(f'Frozen plan mismatch: {key}; use a new run directory.')
        return old
    write_json(target, plan)
    return plan

def verify_plan(plan):
    if sha(plan['dataset']) != plan['dataset_sha256']:
        raise RuntimeError('Dataset changed after freeze')
    for name, expected in plan['source_sha256'].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(f'Code changed after freeze: {name}')
    for item in plan['splits']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Split changed after freeze')

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

def tensor(array, device, dtype=None):
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)

def model_for(method, cfg, fm, fd):
    return create_model(method, fm.shape[1], fd.shape[1], fm.shape[0], fd.shape[0],
                        hidden=cfg['hidden'], layers=cfg['layers'], dropout=cfg['dropout'],
                        alpha=cfg['alpha'], c=cfg['c'])

def score(z, pairs, num_m):
    return (z[pairs[:, 0]] * z[num_m + pairs[:, 1]]).sum(dim=-1) / math.sqrt(64)

@torch.no_grad()
def predict(model, fm, fd, graph, pairs, device):
    model.eval()
    z = model(fm, fd, graph[0], graph[1])
    if not torch.isfinite(z).all():
        raise FloatingPointError('Non-finite evaluation embedding')
    logits = score(z, tensor(pairs, device, torch.long), fm.shape[0])
    return logits.cpu().numpy()

def graph_tensors(context, device):
    return tensor(context['edge_index'], device, torch.long), tensor(context['degree'], device, torch.float32)

def training_seed(seed, epoch, batch):
    return int(np.random.SeedSequence([seed, epoch, batch, 713]).generate_state(1)[0])

def fit_one(split_path, method, config_index, cfg, target, device, smoke_epochs=None):
    target.mkdir(parents=True, exist_ok=True)
    result_file = target / 'validation_result.json'
    if result_file.exists():
        return json.loads(result_file.read_text(encoding='utf-8'))
    split = np.load(split_path, allow_pickle=False)
    # Test arrays are deliberately never accessed in this function.
    seed = int(split['seed'])
    seed_all(seed)
    fm = tensor(split['metabolite_features'], device, torch.float32)
    fd = tensor(split['disease_features'], device, torch.float32)
    nm, nd = fm.shape[0], fd.shape[0]
    train_pos, train_neg = split['train_pos'].copy(), split['train_neg'].copy()
    val_pairs, val_labels = split['val_pairs'].copy(), split['val_labels'].copy()
    train_diseases = split['train_diseases'].copy()
    split.close()
    if len(train_pos) != len(train_neg):
        raise ValueError('Fixed training positives and unknowns must match')
    model = model_for(method, cfg, fm, fd).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    fixed_context = make_context(train_pos, np.empty((0, 2), np.int64), nm, nd, train_diseases, 0)['none']
    fixed_graph = graph_tensors(fixed_context, device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    best, best_epoch, stale = -float('inf'), -1, 0
    max_epochs = smoke_epochs or cfg['max_epochs']
    sync(device)
    start = time.perf_counter()
    log_file = target / 'epochs.jsonl'
    with log_file.open('w', encoding='utf-8') as log:
        for epoch in range(1, max_epochs + 1):
            model.train()
            # Order and graph RNG are independent of model/dropout RNG and config.
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2091]))
            pos_order = rng.permutation(len(train_pos))
            neg_order = rng.permutation(len(train_neg))
            loss_sum, seen, deleted = 0.0, 0, 0
            epoch_start = time.perf_counter()
            for batch, first in enumerate(range(0, len(train_pos), cfg['batch_positive'])):
                pos = train_pos[pos_order[first:first + cfg['batch_positive']]]
                neg = train_neg[neg_order[first:first + len(pos)]]
                context = make_context(train_pos, pos, nm, nd, train_diseases, training_seed(seed, epoch, batch))
                graph = graph_tensors(context[MASK[method]], device)
                queries = tensor(np.concatenate((pos, neg)), device, torch.long)
                labels = torch.cat((torch.ones(len(pos), device=device), torch.zeros(len(neg), device=device)))
                optimizer.zero_grad(set_to_none=True)
                z = model(fm, fd, graph[0], graph[1])
                loss = F.binary_cross_entropy_with_logits(score(z, queries, nm), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at epoch {epoch} batch {batch}')
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(queries)
                seen += len(queries)
                deleted += context['deleted_count'] if MASK[method] != 'none' else 0
            sync(device)
            train_seconds = time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            val_scores = predict(model, fm, fd, fixed_graph, val_pairs, device)
            val_ap = float(validation_macro_ap(val_pairs, val_labels, val_scores, nd))
            if not math.isfinite(val_ap):
                raise FloatingPointError('Validation AP is not finite')
            sync(device)
            val_seconds = time.perf_counter() - val_start
            peak_rss = max(peak_rss, process.memory_info().rss)
            improved = val_ap > best
            if improved:
                best, best_epoch, stale = val_ap, epoch, 0
                checkpoint = dict(method=method, config=cfg, seed=seed, num_m=nm, num_d=nd,
                                  model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                  best_validation_macro_ap=best, best_epoch=best_epoch)
                torch.save(checkpoint, target / 'best.pt')
            else:
                stale += 1
            item = dict(epoch=epoch, loss=loss_sum / seen, validation_macro_ap=val_ap,
                        train_seconds=train_seconds, validation_seconds=val_seconds,
                        additional_deleted_edges_total=deleted, improved=improved)
            log.write(json.dumps(item) + '\n')
            log.flush()
            if epoch == 1 or epoch % 20 == 0:
                print(json.dumps(dict(event='epoch', split=split_path.stem, method=method,
                                      config=config_index, **item)), flush=True)
            if stale >= cfg['patience']:
                break
    sync(device)
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(target / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    val_scores = predict(model, fm, fd, fixed_graph, val_pairs, device)
    metrics = disease_metrics(val_pairs, val_labels, val_scores, nd)
    if not np.isclose(metrics['macro_ap'], best, atol=1e-10, rtol=1e-8):
        raise AssertionError('Saved checkpoint does not reproduce selected validation AP')
    np.savez_compressed(target / 'validation_predictions.npz', pairs=val_pairs, labels=val_labels, logits=val_scores)
    result = dict(split=split_path.stem, seed=seed, method=method, config_index=config_index, configuration=cfg,
                  validation=metrics, best_epoch=best_epoch, epochs_run=epoch,
                  parameters=sum(p.numel() for p in model.parameters()),
                  train_total_seconds=elapsed, peak_process_rss_mib=peak_rss / 2**20,
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
                  checkpoint_sha256=sha(target / 'best.pt'), finished_utc=now(), smoke=bool(smoke_epochs))
    write_json(result_file, result)
    print(json.dumps(dict(event='fit_complete', split=split_path.stem, method=method,
                          config=config_index, best_val_ap=best, epochs=epoch, seconds=elapsed)), flush=True)
    del model, optimizer, fm, fd
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result

def fit(plan, run_dir, device):
    if (run_dir / 'test_evaluation_started.json').exists():
        raise RuntimeError('Formal test evaluation has started; frozen results cannot be tuned further.')
    verify_plan(plan)
    for item in plan['splits']:
        split_path = Path(item['path'])
        for method in plan['methods']:
            configs = plan['configurations'][method]
            results = []
            for index, cfg in enumerate(configs):
                target = run_dir / 'fits' / split_path.stem / method / f'config{index}'
                results.append(fit_one(split_path, method, index, cfg, target, device))
            index = max(range(len(results)), key=lambda i: (results[i]['validation']['macro_ap'], -i))
            write_json(run_dir / 'selections' / split_path.stem / f'{method}.json',
                       dict(split=split_path.stem, method=method, config_index=index,
                            selected_validation_ap=results[index]['validation']['macro_ap'],
                            tuning_total_seconds=sum(r['train_total_seconds'] for r in results),
                            configuration_count=len(results)))
    write_json(run_dir / 'training_complete.json', dict(completed_utc=now(), configurations=120,
                                                       test_arrays_opened=False))

def evaluate(plan, run_dir, device):
    verify_plan(plan)
    if not (run_dir / 'training_complete.json').exists():
        raise RuntimeError('Complete every validation configuration before opening formal test arrays.')
    if not (run_dir / 'test_evaluation_started.json').exists():
        write_json(run_dir / 'test_evaluation_started.json', dict(started_utc=now(), policy='Frozen model and configuration choices'))
    records = []
    for item in plan['splits']:
        split_path = Path(item['path'])
        with np.load(split_path, allow_pickle=False) as split:
            fm = tensor(split['metabolite_features'], device, torch.float32)
            fd = tensor(split['disease_features'], device, torch.float32)
            nm, nd = fm.shape[0], fd.shape[0]
            train_pos, train_diseases = split['train_pos'], split['train_diseases']
            test_pairs, test_labels = split['test_pairs'], split['test_labels']
            fixed = make_context(train_pos, np.empty((0, 2), np.int64), nm, nd, train_diseases, 0)['none']
            graph = graph_tensors(fixed, device)
            for method in plan['methods']:
                selected = json.loads((run_dir / 'selections' / split_path.stem / f'{method}.json').read_text(encoding='utf-8'))
                target = run_dir / 'fits' / split_path.stem / method / f"config{selected['config_index']}"
                test_file = target / 'test_result.json'
                if test_file.exists():
                    records.append(json.loads(test_file.read_text(encoding='utf-8')))
                    continue
                validation = json.loads((target / 'validation_result.json').read_text(encoding='utf-8'))
                if sha(target / 'best.pt') != validation['checkpoint_sha256']:
                    raise RuntimeError('Checkpoint changed after validation selection')
                checkpoint = torch.load(target / 'best.pt', map_location=device, weights_only=True)
                model = model_for(method, checkpoint['config'], fm, fd).to(device)
                model.load_state_dict(checkpoint['model_state'])
                scores = predict(model, fm, fd, graph, test_pairs, device)
                metrics = disease_metrics(test_pairs, test_labels, scores, nd)
                result = dict(validation, test=metrics, tuning_total_seconds=selected['tuning_total_seconds'],
                              test_evaluated_utc=now(), setting=split_path.stem.rsplit('_seed', 1)[0])
                np.savez_compressed(target / 'test_predictions.npz', pairs=test_pairs, labels=test_labels, logits=scores)
                write_json(test_file, result)
                records.append(result)
                print(json.dumps(dict(event='test', split=split_path.stem, method=method, **metrics)), flush=True)
                del model
    write_json(run_dir / 'all_results.json', records)
    write_json(run_dir / 'experiment_complete.json', dict(completed_utc=now(), selected_test_runs=len(records)))
    return records

def report(records, run_dir):
    rows = []
    for setting in ('random_edge', 'cold_disease'):
        for method in METHODS:
            runs = [r for r in records if r['setting'] == setting and r['method'] == method]
            if len(runs) != 3:
                raise AssertionError(f'Expected three seeds: {setting}/{method}')
            row = dict(setting=setting, method=method, n=3)
            for metric in ('macro_ap', 'macro_recall20', 'macro_auroc'):
                values = np.array([r['test'][metric] for r in runs], dtype=float)
                row[metric + '_mean'] = float(values.mean())
                row[metric + '_std'] = float(values.std(ddof=1))
            for metric in ('parameters', 'train_total_seconds', 'peak_cuda_allocated_mib', 'tuning_total_seconds', 'epochs_run'):
                values = [r[metric] for r in runs if r[metric] is not None]
                row[metric + '_mean'] = float(np.mean(values)) if values else None
            row['n_ap_diseases'] = [r['test']['n_ap_diseases'] for r in runs]
            row['n_auroc_diseases'] = [r['test']['n_auroc_diseases'] for r in runs]
            rows.append(row)
    write_json(run_dir / 'summary.json', rows)
    with (run_dir / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# CTD代谢物疾病关联预测：首轮实测结果', '',
             '本报告来自冻结快照与实际训练。每个方法/设定3个重复划分seed，每次2组预先固定验证配置；正式模型测试评分在120组训练及选参全部完成后才执行。独立划分审计检查过测试标签隔离，但未使用模型测试分数。均值±样本标准差（ddof=1）。未标注对暂当负例，指标表示知识库记录回收。', '',
             '随机边与疾病冷启动候选口径不同，不能横向把分数理解成等价难度。冷启动是传导式疾病ID留出，独立属性与节点参与全局注意力。', '']
    for title, methods in [('主实验', MAIN), ('消融实验', ABLATION)]:
        lines += [f'## {title}', '']
        for setting in ('random_edge', 'cold_disease'):
            lines += [f'### {setting}', '', '| 方法 | 宏AP | Recall@20 | 宏AUROC | 参数 | 选中训练秒 | 峰值GPU MiB |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for method in methods:
                r = next(r for r in rows if r['setting'] == setting and r['method'] == method)
                value = lambda key: f"{r[key+'_mean']:.4f} ± {r[key+'_std']:.4f}"
                peak = r['peak_cuda_allocated_mib_mean']
                lines += [f"| {method} | {value('macro_ap')} | {value('macro_recall20')} | {value('macro_auroc')} | {r['parameters_mean']:.0f} | {r['train_total_seconds_mean']:.1f} | {peak:.1f} |" if peak is not None
                          else f"| {method} | {value('macro_ap')} | {value('macro_recall20')} | {value('macro_auroc')} | {r['parameters_mean']:.0f} | {r['train_total_seconds_mean']:.1f} | CPU |"]
            lines.append('')
    lines += ['## 复核与限制', '', '完整结果、各seed有效疾病数、选中配置、实际epoch和调参总耗时见all_results.json与summary.csv。选中训练秒包含其训练、每epoch验证与检查点保存；两组配置总成本另列tuning_total_seconds。峰值为PyTorch分配显存，进程RSS另存，不能等同系统总显存。', '',
              '本轮按事先方案如实保留结果，没有根据测试表现删除或增加模块。结果不保证优于已有五篇论文，不支持患者患病概率或因果推断。领域基线复现与进一步新颖性论证尚未完成。', '',
              '代码来源与差异见../adaptation_notes.md；frozen_plan.json记录数据、代码、划分SHA256及环境。']
    (run_dir / '实验结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'fit', 'evaluate', 'all', 'smoke', 'report'))
    parser.add_argument('--dataset', type=Path, default=HERE / 'data/features/dataset.npz')
    parser.add_argument('--splits', type=Path, default=HERE / 'data/splits')
    parser.add_argument('--run-dir', type=Path, default=HERE / 'runs/20261014_v1')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    if args.stage == 'smoke':
        paths = sorted(args.splits.glob('*_seed17.npz'))
        if not paths:
            raise FileNotFoundError('Generate split files first')
        for method in METHODS:
            fit_one(paths[0], method, 0, settings(method)[0], HERE / 'runs/smoke' / method, device, smoke_epochs=2)
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ('freeze', 'all'):
        splits = sorted(args.splits.glob('*_seed*.npz'))
        if len(splits) != 6:
            raise ValueError('Expected exactly six frozen splits')
        plan = freeze(args.run_dir, args.dataset, splits, device)
    else:
        plan = json.loads((args.run_dir / 'frozen_plan.json').read_text(encoding='utf-8'))
    if args.stage in ('fit', 'all'):
        fit(plan, args.run_dir, device)
    if args.stage in ('evaluate', 'all'):
        report(evaluate(plan, args.run_dir, device), args.run_dir)
    if args.stage == 'report':
        report(json.loads((args.run_dir / 'all_results.json').read_text(encoding='utf-8')), args.run_dir)
    print(json.dumps(dict(event='stage_complete', stage=args.stage, run_dir=str(args.run_dir))), flush=True)

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
