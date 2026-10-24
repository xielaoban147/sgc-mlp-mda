"""Train and select a compact NeurIPS-inspired candidate on validation only.

This script deliberately never indexes `test_pairs` or `test_labels`. The
project's earlier test scores have already been inspected, so this run is
exploratory and cannot provide a fresh test-set claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F

from ego_neighbor_model import EgoNeighborLinkPredictor
from protocol import make_context, disease_metrics, validation_macro_ap

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "revision_v2" / "splits"
BASELINE_RUN = ROOT / "runs" / "20261016_revision_v2"
OUT = ROOT / "runs" / "20261017_ego_neighbor_val_only"
VARIANTS = ("separated_mlp", "ego_only_mlp", "mixed_mlp", "separated_distmult")
LEARNING_RATES = (1e-3, 3e-4)
HIDDEN = 32
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_POSITIVE = 128
DROPOUT = 0.2


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_tensor(array, device, dtype=None):
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph(context, device):
    return _as_tensor(context["edge_index"], device, torch.long)


def _validation(model, fm, fd, edge_index, pairs, labels, num_d, device):
    model.eval()
    with torch.no_grad():
        logits = model(fm, fd, edge_index, _as_tensor(pairs, device, torch.long)).cpu().numpy()
    metrics = disease_metrics(pairs, labels, logits, num_d)
    return logits, metrics


def _fit_one(split_path: Path, variant: str, lr: float, device):
    seed_text = split_path.stem
    target = OUT / "fits" / seed_text / variant / f"lr{lr:g}"
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / "validation_result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    # Only training and validation fields are read; the test arrays are not indexed.
    with np.load(split_path, allow_pickle=False) as data:
        seed = int(data["seed"])
        fm = _as_tensor(data["metabolite_features"], device, torch.float32)
        fd = _as_tensor(data["disease_features"], device, torch.float32)
        train_pos = data["train_pos"].copy()
        train_neg = data["train_neg"].copy()
        val_pairs = data["val_pairs"].copy()
        val_labels = data["val_labels"].copy()
        train_diseases = data["train_diseases"].copy()
    num_m, num_d = fm.shape[0], fd.shape[0]
    _seed_all(seed)
    model = EgoNeighborLinkPredictor(fm.shape[1], fd.shape[1], num_m, num_d,
                                     hidden=HIDDEN, dropout=DROPOUT,
                                     variant=variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    fixed_context = make_context(train_pos, np.empty((0, 2), np.int64),
                                 num_m, num_d, train_diseases, 0)["none"]
    fixed_edge_index = _graph(fixed_context, device)
    best_ap, best_epoch, stale = -float("inf"), -1, 0
    log_path = target / "epochs.jsonl"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2091]))
            pos_order = rng.permutation(len(train_pos))
            neg_order = rng.permutation(len(train_neg))
            loss_sum = seen = 0
            for batch, first in enumerate(range(0, len(train_pos), BATCH_POSITIVE)):
                pos = train_pos[pos_order[first:first + BATCH_POSITIVE]]
                neg = train_neg[neg_order[first:first + len(pos)]]
                context = make_context(train_pos, pos, num_m, num_d, train_diseases,
                                       int(np.random.SeedSequence([seed, epoch, batch, 713]).generate_state(1)[0]))["none"]
                edge_index = _graph(context, device)
                pairs = _as_tensor(np.concatenate((pos, neg)), device, torch.long)
                labels = torch.cat((torch.ones(len(pos), device=device),
                                    torch.zeros(len(pos), device=device)))
                optimizer.zero_grad(set_to_none=True)
                logits = model(fm, fd, edge_index, pairs)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: {seed_text}/{variant}/{lr}/epoch{epoch}")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(pairs)
                seen += len(pairs)
            logits, metrics = _validation(model, fm, fd, fixed_edge_index,
                                          val_pairs, val_labels, num_d, device)
            val_ap = float(validation_macro_ap(val_pairs, val_labels, logits, num_d))
            if not np.isclose(val_ap, metrics["macro_ap"], rtol=1e-8, atol=1e-10):
                raise AssertionError("Validation AP implementations disagree")
            improved = val_ap > best_ap
            if improved:
                best_ap, best_epoch, stale = val_ap, epoch, 0
                torch.save({"variant": variant, "lr": lr, "seed": seed,
                            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "best_validation_macro_ap": best_ap, "best_epoch": best_epoch}, target / "best.pt")
                np.savez_compressed(target / "best_validation_predictions.npz",
                                    pairs=val_pairs, labels=val_labels, logits=logits)
            else:
                stale += 1
            log.write(json.dumps({"epoch": epoch, "train_loss": loss_sum / seen,
                                  "validation_macro_ap": val_ap, "improved": improved}) + "\n")
            if epoch == 1 or epoch % 20 == 0:
                print(json.dumps({"event": "epoch", "split": seed_text,
                                  "variant": variant, "lr": lr, "epoch": epoch,
                                  "validation_ap": val_ap}), flush=True)
            if stale >= PATIENCE:
                break
    elapsed = time.perf_counter() - start
    model.load_state_dict(torch.load(target / "best.pt", map_location=device,
                                     weights_only=True)["state_dict"])
    _, best_metrics = _validation(model, fm, fd, fixed_edge_index,
                                  val_pairs, val_labels, num_d, device)
    result = {"split": seed_text, "seed": seed, "variant": variant, "lr": lr,
              "hidden": HIDDEN, "dropout": DROPOUT, "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": best_epoch, "epochs_run": epoch,
              "validation": best_metrics, "seconds": elapsed,
              "checkpoint_sha256": _sha(target / "best.pt"),
              "test_arrays_opened": False}
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "fit_complete", "split": seed_text, "variant": variant,
                      "lr": lr, "best_validation_ap": best_ap, "epoch": best_epoch}), flush=True)
    return result


def _baseline_result(split_name: str, method: str):
    selection_path = BASELINE_RUN / "selections" / split_name / f"{method}.json"
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    config_dir = BASELINE_RUN / "fits" / split_name / method / f"config{selected['config_index']}"
    result = json.loads((config_dir / "validation_result.json").read_text(encoding="utf-8"))
    return result


def _baseline_validation(split_name: str, method: str):
    return _baseline_result(split_name, method)["validation"]["macro_ap"]


def _baseline_parameters(split_name: str, method: str):
    return _baseline_result(split_name, method)["parameters"]


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    split_paths = sorted(SPLIT_DIR.glob("*_seed*.npz"))
    if len(split_paths) != 10:
        raise RuntimeError(f"Expected 10 pre-existing exploratory splits; found {len(split_paths)}")
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "purpose": "validation-only exploratory candidate and ablation study",
                "inspiration": "Zhu et al., NeurIPS 2024: ego/neighbor separation + learnable decoder",
                "source_paper": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/79353864175b6b8c3d073cde84d7014a-Abstract-Conference.html",
                "source_code": "https://github.com/GemsLab/HeteLinkPred",
                "split_hashes": {p.name: _sha(p) for p in split_paths},
                "source_hashes": {name: _sha(ROOT / name) for name in
                                  ("models.py", "protocol.py", "ego_neighbor_model.py", "run_ego_neighbor_validation.py")},
                "variants": list(VARIANTS), "learning_rates": list(LEARNING_RATES),
                "hidden": HIDDEN, "dropout": DROPOUT, "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE, "test_arrays_opened": False,
                "important_limit": "Old tests were already inspected; no independent test claim is possible from these splits."}
    (OUT / "validation_only_plan.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    results = []
    for split_path in split_paths:
        for variant in VARIANTS:
            for lr in LEARNING_RATES:
                results.append(_fit_one(split_path, variant, lr, device))
    (OUT / "validation_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = []
    for setting in ("random_edge", "cold_disease"):
        for variant in VARIANTS:
            subset = [r for r in results if r["split"].startswith(setting) and r["variant"] == variant]
            selected_by_split = {}
            for split_path in [p for p in split_paths if p.stem.startswith(setting)]:
                options = [r for r in subset if r["split"] == split_path.stem]
                selected_by_split[split_path.stem] = max(options, key=lambda r: (r["validation"]["macro_ap"], -r["lr"]))
            aps = np.asarray([v["validation"]["macro_ap"] for v in selected_by_split.values()])
            summary.append({"setting": setting, "method": variant, "n": len(aps),
                            "validation_ap_mean": float(aps.mean()), "validation_ap_std": float(aps.std(ddof=1)),
                            "selected_lr_by_split": {k: v["lr"] for k, v in selected_by_split.items()},
                            "selected_validation_ap_by_split": {k: v["validation"]["macro_ap"] for k, v in selected_by_split.items()},
                            "parameters_mean": float(np.mean([v["parameters"] for v in selected_by_split.values()]))})
    for setting in ("random_edge", "cold_disease"):
        for method in ("sgc", "fixed_tuned", "residual_fixed_budget", "residual_budget"):
            vals = [_baseline_validation(p.stem, method) for p in split_paths if p.stem.startswith(setting)]
            summary.append({"setting": setting, "method": method + " (previous validation selected)", "n": len(vals),
                            "validation_ap_mean": float(np.mean(vals)), "validation_ap_std": float(np.std(vals, ddof=1)),
                            "selected_validation_ap_by_split": {p.stem: _baseline_validation(p.stem, method)
                                                                 for p in split_paths if p.stem.startswith(setting)}})
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Ego-neighbor轻量候选：验证集探索结果", "",
             "灵感来自NeurIPS 2024关于特征异配链路预测的分析：节点自身表示与邻居表示分开处理，并使用可学习解码器。这里是小型代谢物-疾病任务适配。", "",
             "本轮只读取训练字段与验证字段。旧正式测试结果已在前轮查看，因此本结果只能做验证集筛选和机制诊断，不能当作独立测试、显著性证明或泛化结论。", "",
             "模型：共享类型特征前端；32维ego映射；32维邻居均值映射；MLP解码器对端点状态、逐元素乘积、绝对差建模。四种结构在两档学习率下比较，dropout=0.2，200 epoch上限，验证AP早停。", "",
             "| 场景 | 方法 | 验证宏AP | 参数数 |", "|---|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['setting']} | {row['method']} | {row['validation_ap_mean']:.4f} ± {row['validation_ap_std']:.4f} | {row.get('parameters_mean', float('nan')):.0f} |")
    lines += ["", "按场景列出各划分值，避免只看均值决定：", ""]
    for row in summary:
        lines.append(f"- {row['setting']} / {row['method']}: " + ", ".join(f"{k.rsplit('_seed', 1)[-1]}={v:.4f}" for k, v in row["selected_validation_ap_by_split"].items()))
    lines += ["", "结果文件保存每个epoch日志、最优验证checkpoint及预测分数。测试数组未被读取。"]
    (OUT / "验证实验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "out": str(OUT), "fits": len(results), "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
