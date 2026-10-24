"""Predeclared interaction-feature ablations on the ten existing CTD splits.

No existing source, checkpoint, split, or frozen result is changed. The full
decoder's original frozen evaluation is reused after checkpoint/output parity
checks; three zero-masked feature ablations are fitted on validation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUPPORT = HERE / "frozen_support"
SPLITS = ROOT / "revision_v2" / "splits"
OLD_VAL = ROOT / "runs" / "20261017_sgc_mlp_val_only"
OLD_TEST = ROOT / "runs" / "20261017_sgc_mlp_test_once"
VARIANTS = ("endpoint_only", "endpoint_product", "endpoint_absdiff")
ALL_VARIANTS = (*VARIANTS, "full")
LR = (0.001, 0.0003)
SUPPORT_NAMES = ("models.py", "protocol.py", "ego_neighbor_model.py",
                 "run_ego_neighbor_validation.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value):
    if path.exists():
        raise FileExistsError(f"Immutable record already exists: {path}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_runtime():
    sys.path.insert(0, str(SUPPORT))
    import numpy as np
    import torch
    import run_ego_neighbor_validation as runner
    from ego_neighbor_model import EgoNeighborLinkPredictor

    class MaskedInteractionModel(EgoNeighborLinkPredictor):
        def __init__(self, metabolite_dim, disease_dim, num_m, num_d,
                     hidden=64, dropout=0.2, variant="full"):
            if variant not in ALL_VARIANTS:
                raise ValueError(variant)
            super().__init__(metabolite_dim, disease_dim, num_m, num_d,
                             hidden=hidden, dropout=dropout, variant="sgc_mlp")
            self.feature_set = variant

        def _pair_features(self, left, right):
            product = left * right
            difference = torch.abs(left - right)
            if self.feature_set in ("endpoint_only", "endpoint_absdiff"):
                product = torch.zeros_like(product)
            if self.feature_set in ("endpoint_only", "endpoint_product"):
                difference = torch.zeros_like(difference)
            return torch.cat((left, right, product, difference), dim=-1)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.OUT = HERE / "validation"
    runner.HIDDEN = 64
    runner.EgoNeighborLinkPredictor = MaskedInteractionModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return np, torch, runner, MaskedInteractionModel, device


def prepare():
    plan_path = HERE / "frozen_ablation_plan.json"
    if plan_path.exists():
        raise FileExistsError(plan_path)
    SUPPORT.mkdir(parents=True, exist_ok=False)
    for name in SUPPORT_NAMES:
        shutil.copy2(ROOT / name, SUPPORT / name)
    paths = sorted(SPLITS.glob("*_seed*.npz"))
    if len(paths) != 10:
        raise RuntimeError("Exactly ten fixed splits are required")
    old_freeze = read(OLD_VAL / "validation_only_plan.json")
    plan = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "reviewer-requested controlled decoder-feature ablation; all predeclared variants reported",
        "training_variants": list(VARIANTS), "comparison_variants": list(ALL_VARIANTS),
        "feature_blocks": "[z_m(64), z_d(64), product(64), absolute_difference(64)]",
        "ablation_control": "Removed blocks are replaced with exact zeros; encoder, 256-64-1 decoder allocation, initialization, dropout and optimizer are shared",
        "nominal_trainable_parameters": 484033,
        "inactive_first_layer_weights": {"endpoint_only": 8192, "endpoint_product": 4096, "endpoint_absdiff": 4096, "full": 0},
        "learning_rates": list(LR), "learning_rate_tie_break": "highest validation macro-AP; ties choose lower learning rate",
        "max_epochs": 200, "patience": 20, "dropout": 0.2, "positive_batch_size": 128,
        "negative_sampling": "Use the same fixed train_neg pool stored in each split (1:1 with train_pos); shuffle both pools independently every epoch using SeedSequence([seed,epoch,2091]); no replacement or test-label tuning",
        "graph_context": "Remove all supervised positive queries in the current batch from train_pos in both directions; no additional context deletion; validation/test use the complete training-positive graph",
        "selection": "Select checkpoint epoch and learning rate exclusively by validation macro-AP, independently for every variant/split",
        "test_rule": "Freeze all 30 validation selections and checkpoint hashes before opening new test arrays; evaluate each selected checkpoint once; retain every variant regardless of test ranking; no change to model, seed, or learning rates after test",
        "reuse_full": "Reuse original ten full-decoder frozen test results only after exact checkpoint hashes, split hashes and bit-exact stored validation-prediction parity checks",
        "test_history": "These are the previously explored five-seed splits, not a new independently held-out cohort; this revision is a mechanism ablation",
        "split_sha256": {p.stem: sha(p) for p in paths},
        "support_sha256": {name: sha(SUPPORT / name) for name in SUPPORT_NAMES},
        "script_sha256": sha(Path(__file__)),
        "old_validation_source_hash_match": {name: sha(SUPPORT / name) == old_freeze["source_hashes"][name] for name in SUPPORT_NAMES},
        "old_test_results_sha256": sha(OLD_TEST / "test_results.json"),
        "old_test_plan_sha256": sha(OLD_TEST / "test_evaluation_plan.json"),
        "threads": 2,
    }
    write_once(plan_path, plan)
    print(json.dumps({"event": "plan_frozen", "path": str(plan_path)}), flush=True)


def verify_plan():
    plan = read(HERE / "frozen_ablation_plan.json")
    assert plan["script_sha256"] == sha(Path(__file__))
    assert all(sha(SUPPORT / name) == digest for name, digest in plan["support_sha256"].items())
    assert all(sha(SPLITS / f"{name}.npz") == digest for name, digest in plan["split_sha256"].items())
    assert sha(OLD_TEST / "test_results.json") == plan["old_test_results_sha256"]
    return plan


def verify_full(np, torch, runner, Model, device):
    if (HERE / "full_reuse_parity_audit.json").exists():
        return
    old_plan = read(OLD_TEST / "test_evaluation_plan.json")
    checks = []
    for split_name, lr in old_plan["selected_learning_rate_by_split"].items():
        split_path = SPLITS / f"{split_name}.npz"
        checkpoint = OLD_VAL / "fits" / split_name / "sgc_mlp" / f"lr{lr:g}" / "best.pt"
        assert sha(split_path) == old_plan["split_sha256"][split_name]
        assert sha(checkpoint) == old_plan["checkpoint_sha256"][split_name]
        with np.load(split_path, allow_pickle=False) as d:
            fm = torch.as_tensor(d["metabolite_features"], dtype=torch.float32, device=device)
            fd = torch.as_tensor(d["disease_features"], dtype=torch.float32, device=device)
            pos, diseases = d["train_pos"].copy(), d["train_diseases"].copy()
            pairs, labels = d["val_pairs"].copy(), d["val_labels"].copy()
        model = Model(fm.shape[1], fd.shape[1], len(fm), len(fd), variant="full").to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        ctx = runner.make_context(pos, np.empty((0, 2), dtype=np.int64), len(fm), len(fd), diseases, 0)["none"]
        logits, metrics = runner._validation(model, fm, fd, runner._graph(ctx, device), pairs, labels, len(fd), device)
        with np.load(checkpoint.parent / "best_validation_predictions.npz", allow_pickle=False) as old:
            bit_exact = bool(np.array_equal(old["logits"], logits))
            maximum_difference = float(np.max(np.abs(old["logits"] - logits)))
            assert np.array_equal(old["pairs"], pairs) and np.array_equal(old["labels"], labels)
        if not bit_exact:
            raise AssertionError(f"Full implementation parity failed: {split_name}: {maximum_difference}")
        checks.append({"split": split_name, "checkpoint_sha256": sha(checkpoint), "validation_predictions_bit_exact": bit_exact,
                       "maximum_absolute_logit_difference": maximum_difference, "validation_macro_ap": metrics["macro_ap"]})
    write_once(HERE / "full_reuse_parity_audit.json", {"all_ten_bit_exact": True, "checks": checks,
                "test_arrays_opened": False, "models_and_protocol_unchanged_since_original_plan": True,
                "later_source_changes": "ego model and runner sources have added non-SGC variants since the original plan; SGC checkpoint loading and all stored validation logits are bit-exact"})
    print(json.dumps({"event": "full_reuse_verified", "bit_exact_splits": len(checks)}), flush=True)


def train(np, torch, runner, Model, device):
    verify_plan()
    verify_full(np, torch, runner, Model, device)
    if (HERE / "frozen_test_selection.json").exists():
        print("Test selections already frozen; training will not run.", flush=True)
        return
    results = []
    for split_path in sorted(SPLITS.glob("*_seed*.npz")):
        for variant in VARIANTS:
            for lr in LR:
                results.append(runner._fit_one(split_path, variant, lr, device))
    write_once(HERE / "validation_results.json", results)
    selected = []
    for split_path in sorted(SPLITS.glob("*_seed*.npz")):
        for variant in VARIANTS:
            best = max((r for r in results if r["split"] == split_path.stem and r["variant"] == variant),
                       key=lambda r: (r["validation"]["macro_ap"], -r["lr"]))
            checkpoint = HERE / "validation" / "fits" / best["split"] / variant / f"lr{best['lr']:g}" / "best.pt"
            assert sha(checkpoint) == best["checkpoint_sha256"]
            selected.append({"split": best["split"], "variant": variant, "lr": best["lr"], "best_epoch": best["best_epoch"],
                             "validation": best["validation"], "parameters": best["parameters"],
                             "checkpoint": str(checkpoint.relative_to(HERE)), "checkpoint_sha256": sha(checkpoint)})
    write_once(HERE / "frozen_test_selection.json", {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "selection_rule": "validation macro-AP, lower-LR tie break", "test_arrays_opened": False,
                 "selections": selected, "ablation_plan_sha256": sha(HERE / "frozen_ablation_plan.json")})
    print(json.dumps({"event": "all_validation_selections_frozen", "n": len(selected)}), flush=True)


def test(np, torch, runner, Model, device):
    verify_plan()
    freeze = read(HERE / "frozen_test_selection.json")
    out = HERE / "test_once"
    out.mkdir(exist_ok=False)
    write_once(out / "evaluation_started.json", {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "selection_freeze_sha256": sha(HERE / "frozen_test_selection.json"), "n": 30})
    results = []
    for choice in freeze["selections"]:
        checkpoint = HERE / choice["checkpoint"]
        assert sha(checkpoint) == choice["checkpoint_sha256"]
        with np.load(SPLITS / f"{choice['split']}.npz", allow_pickle=False) as d:
            fm = torch.as_tensor(d["metabolite_features"], dtype=torch.float32, device=device)
            fd = torch.as_tensor(d["disease_features"], dtype=torch.float32, device=device)
            pos, diseases = d["train_pos"].copy(), d["train_diseases"].copy()
            pairs, labels = d["test_pairs"].copy(), d["test_labels"].copy()
        model = Model(fm.shape[1], fd.shape[1], len(fm), len(fd), variant=choice["variant"]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
        ctx = runner.make_context(pos, np.empty((0, 2), dtype=np.int64), len(fm), len(fd), diseases, 0)["none"]
        logits, metrics = runner._validation(model, fm, fd, runner._graph(ctx, device), pairs, labels, len(fd), device)
        record = {**choice, "test": metrics, "evaluation_source": "new single locked-checkpoint evaluation"}
        results.append(record)
        np.savez_compressed(out / f"{choice['split']}_{choice['variant']}_predictions.npz", pairs=pairs, labels=labels, logits=logits)
        write_once(out / f"{choice['split']}_{choice['variant']}_result.json", record)
        print(json.dumps({"event": "test_result", "split": choice["split"], "variant": choice["variant"], "test": metrics}), flush=True)
    for old in read(OLD_TEST / "test_results.json"):
        results.append({**old, "variant": "full", "evaluation_source": "reused original frozen full-decoder test result; no new evaluation"})
    write_once(HERE / "test_results_all_variants.json", results)
    summarize(np)


def summarize(np):
    results = read(HERE / "test_results_all_variants.json")
    summary = []
    comparisons = []
    for setting in ("random_edge", "cold_disease"):
        for variant in ALL_VARIANTS:
            rows = [r for r in results if r["split"].startswith(setting) and r["variant"] == variant]
            assert len(rows) == 5
            row = {"setting": setting, "variant": variant, "n": 5, "nominal_parameters": 484033, "metrics": {}}
            for metric in ("macro_ap", "macro_recall20", "macro_auroc"):
                vals = np.array([r["test"][metric] for r in rows])
                row["metrics"][metric] = {"mean": float(vals.mean()), "sample_std": float(vals.std(ddof=1)),
                                          "per_split": {r["split"]: r["test"][metric] for r in rows}}
            summary.append(row)
        full = {r["split"]: r for r in results if r["split"].startswith(setting) and r["variant"] == "full"}
        for variant in VARIANTS:
            rows = [r for r in results if r["split"].startswith(setting) and r["variant"] == variant]
            delta = {r["split"]: full[r["split"]]["test"]["macro_ap"] - r["test"]["macro_ap"] for r in rows}
            comparisons.append({"setting": setting, "full_minus_variant": variant, "macro_ap_mean_delta": float(np.mean(list(delta.values()))),
                                "sample_std_of_delta": float(np.std(list(delta.values()), ddof=1)),
                                "positive_delta_seeds": int(sum(x > 0 for x in delta.values())), "per_split_delta": delta})
    write_once(HERE / "ablation_summary.json", summary)
    write_once(HERE / "paired_comparisons.json", comparisons)
    lines = ["# 2026-10-23 decoder-feature ablation", "", "Five pre-existing seeds per setting. Values are mean ± sample standard deviation (ddof=1).", "",
             "The 256-dimensional decoder input and 484,033 allocated trainable parameters are identical. Removed blocks are set to zero; their first-layer columns receive no gradient. Full results are reused from the original frozen evaluation after bit-exact validation-logit parity checks.", "",
             "| Setting | Feature set | Macro AP | Recall@20 | AUROC |", "|---|---|---:|---:|---:|"]
    for row in summary:
        cells = [f"{row['metrics'][m]['mean']:.4f} ± {row['metrics'][m]['sample_std']:.4f}" for m in ("macro_ap", "macro_recall20", "macro_auroc")]
        lines.append(f"| {row['setting']} | {row['variant']} | " + " | ".join(cells) + " |")
    lines += ["", "Paired macro-AP differences (full minus variant):", ""]
    for row in comparisons:
        lines.append(f"- {row['setting']} / {row['full_minus_variant']}: {row['macro_ap_mean_delta']:+.4f}; positive in {row['positive_delta_seeds']}/5 seeds.")
    lines += ["", "No test-based variant, epoch, seed, or learning-rate selection was performed in this ablation. These existing splits have been used in earlier model development; this is a controlled mechanism study, not an untouched external evaluation.", "",
              "Reproduction from experiments/ctd_sgformer (use a fresh output copy if the immutable records already exist):", "", "```powershell", ".\\.venv\\Scripts\\python.exe reviewer_revision_20261023/run_decoder_ablation.py --phase prepare", ".\\.venv\\Scripts\\python.exe reviewer_revision_20261023/run_decoder_ablation.py --phase train", ".\\.venv\\Scripts\\python.exe reviewer_revision_20261023/run_decoder_ablation.py --phase test", "```", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(HERE), "rows": len(results)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "train", "test"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        runtime = load_runtime()
        (train if args.phase == "train" else test)(*runtime)


if __name__ == "__main__":
    main()
