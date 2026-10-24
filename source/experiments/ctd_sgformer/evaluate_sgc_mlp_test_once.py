"""Evaluate the validation-selected SGC+interaction-MLP checkpoints once.

The architecture, per-split learning rate, and checkpoint epoch are fixed by
the validation-only run. This script reads test pairs and labels only after
those choices are frozen and does not fit or tune on test data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from ego_neighbor_model import EgoNeighborLinkPredictor
from protocol import disease_metrics, make_context

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "revision_v2" / "splits"
VALIDATION_RUN = ROOT / "runs" / "20261017_sgc_mlp_val_only"
OUT = ROOT / "runs" / "20261017_sgc_mlp_test_once"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen test evaluation: {OUT}")
    OUT.mkdir(parents=True)
    validation_summary = json.loads(
        (VALIDATION_RUN / "validation_summary.json").read_text(encoding="utf-8")
    )
    selected = {}
    for row in validation_summary:
        if row["method"] == "SGC + interaction MLP":
            selected.update(row["selected_lr_by_split"])
    split_paths = sorted(SPLIT_DIR.glob("*_seed*.npz"))
    if len(split_paths) != 10 or set(selected) != {p.stem for p in split_paths}:
        raise RuntimeError("Expected validation-selected learning rates for all 10 fixed splits")

    plan = {
        "purpose": "single post-selection evaluation of the locked SGC+interaction-MLP candidate",
        "selection_source": str(VALIDATION_RUN / "validation_summary.json"),
        "architecture": "type-specific feature adapter + one-hop SGC + 256-64-1 interaction MLP",
        "decoder_features": "[z_m, z_d, z_m*z_d, abs(z_m-z_d)]",
        "hidden": 64,
        "dropout": 0.2,
        "selected_learning_rate_by_split": selected,
        "test_tuning": False,
        "split_sha256": {p.stem: sha256(p) for p in split_paths},
        "checkpoint_sha256": {},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    results = []
    for split_path in split_paths:
        name = split_path.stem
        lr = selected[name]
        checkpoint = VALIDATION_RUN / "fits" / name / "sgc_mlp" / f"lr{lr:g}" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        plan["checkpoint_sha256"][name] = sha256(checkpoint)

        with np.load(split_path, allow_pickle=False) as split:
            fm = torch.as_tensor(split["metabolite_features"], dtype=torch.float32, device=device)
            fd = torch.as_tensor(split["disease_features"], dtype=torch.float32, device=device)
            train_pos = split["train_pos"].copy()
            train_diseases = split["train_diseases"].copy()
            test_pairs = split["test_pairs"].copy()
            test_labels = split["test_labels"].copy()
        num_m, num_d = fm.shape[0], fd.shape[0]
        context = make_context(train_pos, np.empty((0, 2), dtype=np.int64),
                               num_m, num_d, train_diseases, 0)["none"]
        edge_index = torch.as_tensor(context["edge_index"], dtype=torch.long, device=device)
        model = EgoNeighborLinkPredictor(
            fm.shape[1], fd.shape[1], num_m, num_d,
            hidden=64, dropout=0.2, variant="sgc_mlp"
        ).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(
                fm, fd, edge_index,
                torch.as_tensor(test_pairs, dtype=torch.long, device=device)
            ).cpu().numpy()
        metrics = disease_metrics(test_pairs, test_labels, logits, num_d)
        result = {
            "split": name,
            "learning_rate_selected_on_validation": lr,
            "best_epoch_selected_on_validation": int(saved["best_epoch"]),
            "parameters": sum(p.numel() for p in model.parameters()),
            "test": metrics,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    for setting in ("random_edge", "cold_disease"):
        rows = [r for r in results if r["split"].startswith(setting)]
        for metric in ("macro_ap", "macro_recall20", "macro_auroc"):
            values = np.asarray([r["test"][metric] for r in rows], dtype=np.float64)
            plan.setdefault("summary", {}).setdefault(setting, {})[metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "per_split": {r["split"]: r["test"][metric] for r in rows},
            }

    (OUT / "test_evaluation_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "test_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output": str(OUT),
                      "device": str(device), "n_splits": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
