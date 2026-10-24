"""Test a small nonlinear decoder on the strongest existing SGC backbone.

Only train and validation arrays are indexed. The earlier formal test set has
already been inspected, so this remains exploratory model development.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import run_ego_neighbor_validation as runner

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "20261017_sgc_mlp_val_only"


def main():
    runner.OUT = OUT
    runner.HIDDEN = 64
    split_paths = sorted(runner.SPLIT_DIR.glob("*_seed*.npz"))
    if len(split_paths) != 10:
        raise RuntimeError(f"Expected 10 pre-existing exploratory splits, got {len(split_paths)}")
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "purpose": "validation-only development of an MLP pairwise decoder on the SGC backbone",
            "inspiration": "Zhu et al., NeurIPS 2024; nonlinear MLP decoder for non-homophilic/gated link prediction",
            "source_paper": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/79353864175b6b8c3d073cde84d7014a-Abstract-Conference.html",
            "source_code": "https://github.com/GemsLab/HeteLinkPred",
            "architecture": "shared type-specific input adapter + one-hop SGC encoder + 256-64-1 MLP on endpoint, product, and absolute-difference features",
            "hidden": 64, "decoder_hidden": 64, "dropout": runner.DROPOUT,
            "learning_rates": list(runner.LEARNING_RATES), "max_epochs": runner.MAX_EPOCHS,
            "patience": runner.PATIENCE,
            "split_hashes": {p.name: runner._sha(p) for p in split_paths},
            "source_hashes": {name: runner._sha(ROOT / name) for name in
                              ("models.py", "protocol.py", "ego_neighbor_model.py",
                               "run_ego_neighbor_validation.py", "run_sgc_mlp_validation.py")},
            "test_arrays_opened": False,
            "limitation": "Prior test scores were inspected in an earlier experiment; this run cannot establish an independent test-set win."}
    plan_path = OUT / "validation_only_plan.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    results = []
    device = runner.torch.device("cuda" if runner.torch.cuda.is_available() else "cpu")
    for split_path in split_paths:
        for lr in runner.LEARNING_RATES:
            results.append(runner._fit_one(split_path, "sgc_mlp", lr, device))
    (OUT / "validation_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = []
    for setting in ("random_edge", "cold_disease"):
        candidate = {r["split"]: r for r in results if r["split"].startswith(setting)}
        selected = {split: max((r for r in results if r["split"] == split),
                               key=lambda r: (r["validation"]["macro_ap"], -r["lr"]))
                    for split in candidate}
        vals = np.asarray([r["validation"]["macro_ap"] for r in selected.values()])
        summary.append({"setting": setting, "method": "SGC + interaction MLP", "n": len(vals),
                        "validation_ap_mean": float(vals.mean()), "validation_ap_std": float(vals.std(ddof=1)),
                        "selected_lr_by_split": {k: v["lr"] for k, v in selected.items()},
                        "selected_validation_ap_by_split": {k: v["validation"]["macro_ap"] for k, v in selected.items()},
                        "parameters_mean": float(np.mean([v["parameters"] for v in selected.values()]))})
        for method in ("sgc", "fixed_tuned", "residual_fixed_budget", "residual_budget"):
            vals = {p.stem: runner._baseline_validation(p.stem, method)
                    for p in split_paths if p.stem.startswith(setting)}
            params = [runner._baseline_parameters(p.stem, method)
                      for p in split_paths if p.stem.startswith(setting)]
            summary.append({"setting": setting, "method": method + " (previous validation selected)", "n": len(vals),
                            "validation_ap_mean": float(np.mean(list(vals.values()))),
                            "validation_ap_std": float(np.std(list(vals.values()), ddof=1)),
                            "parameters_mean": float(np.mean(params)),
                            "selected_validation_ap_by_split": vals})
    paired = []
    for setting in ("random_edge", "cold_disease"):
        candidate = next(r for r in summary if r["setting"] == setting and r["method"] == "SGC + interaction MLP")
        for method in ("sgc", "fixed_tuned", "residual_fixed_budget", "residual_budget"):
            baseline = next(r for r in summary if r["setting"] == setting and r["method"].startswith(method + " "))
            deltas = {key: candidate["selected_validation_ap_by_split"][key] - value
                      for key, value in baseline["selected_validation_ap_by_split"].items()}
            paired.append({"setting": setting, "candidate": "SGC + interaction MLP", "baseline": method,
                           "mean_validation_ap_delta": float(np.mean(list(deltas.values()))),
                           "wins": int(sum(value > 0 for value in deltas.values())),
                           "n": len(deltas), "per_split_delta": deltas})
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "paired_validation_comparisons.json").write_text(json.dumps(paired, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# SGC + 非线性交互解码器：验证集结果", "",
             "候选在原 SGC 编码器后增加一个轻量 MLP 配对解码器，输入两端嵌入、逐元素乘积和绝对差。该组合直接检验 NeurIPS 2024 链路预测工作提出的可学习解码器思路。", "",
             "由于更早的实验已经看过这些划分的测试表现，本表是验证集开发结果，不能作为独立测试或正式泛化结论。", "",
             "| 场景 | 方法 | 验证宏AP | 参数量 |", "|---|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['setting']} | {row['method']} | {row['validation_ap_mean']:.4f} ± {row['validation_ap_std']:.4f} | {row.get('parameters_mean', 0):.0f} |")
    lines += ["", "逐seed验证AP：", ""]
    for row in summary:
        lines.append(f"- {row['setting']} / {row['method']}: " + ", ".join(
            f"{k.rsplit('_seed', 1)[-1]}={v:.4f}" for k, v in row["selected_validation_ap_by_split"].items()))
    lines += ["", "候选减基线的配对验证AP：", ""]
    for row in paired:
        delta = row["mean_validation_ap_delta"]
        lines.append(f"- {row['setting']}，对 {row['baseline']}：均差 {delta:+.4f}，胜 {row['wins']}/{row['n']}；逐折 " +
                     ", ".join(f"{k.rsplit('_seed', 1)[-1]}={v:+.4f}" for k, v in row["per_split_delta"].items()))
    lines += ["", "训练脚本只索引 train/validation 字段；不读取 `test_pairs` 或 `test_labels`。"]
    (OUT / "验证实验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "out": str(OUT), "fits": len(results), "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
