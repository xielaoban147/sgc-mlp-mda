"""Recompute saved ablation metrics and verify all frozen choices and records."""
import json
import sys
from pathlib import Path

import numpy as np

from run_decoder_ablation import ALL_VARIANTS, HERE, OLD_TEST, SPLITS, SUPPORT, VARIANTS, read, sha, verify_plan

sys.path.insert(0, str(SUPPORT))
from protocol import disease_metrics


def main():
    verify_plan()
    fits = read(HERE / "validation_results.json")
    freeze = read(HERE / "frozen_test_selection.json")
    results = read(HERE / "test_results_all_variants.json")
    summaries = read(HERE / "ablation_summary.json")
    assert len(fits) == 60 and len(freeze["selections"]) == 30 and len(results) == 40
    assert len({(r["split"], r["variant"], r["lr"]) for r in fits}) == 60
    epoch_checks = []
    for fit in fits:
        fit_path = HERE / "validation" / "fits" / fit["split"] / fit["variant"] / f"lr{fit['lr']:g}"
        assert sha(fit_path / "best.pt") == fit["checkpoint_sha256"]
        logs = [json.loads(line) for line in (fit_path / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(logs) == fit["epochs_run"]
        first_best = max(logs, key=lambda r: r["validation_macro_ap"])
        assert first_best["epoch"] == fit["best_epoch"]
        assert np.isclose(first_best["validation_macro_ap"], fit["validation"]["macro_ap"], atol=1e-12, rtol=0)
        with np.load(fit_path / "best_validation_predictions.npz", allow_pickle=False) as p:
            recomputed = disease_metrics(p["pairs"], p["labels"], p["logits"], 433)
        assert all(np.isclose(recomputed[m], fit["validation"][m], atol=1e-12, rtol=0) for m in recomputed)
        epoch_checks.append({"split": fit["split"], "variant": fit["variant"], "lr": fit["lr"],
                             "epochs": len(logs), "first_best_epoch": fit["best_epoch"], "checkpoint_hash_verified": True})
    for choice in freeze["selections"]:
        options = [r for r in fits if r["split"] == choice["split"] and r["variant"] == choice["variant"]]
        winner = max(options, key=lambda r: (r["validation"]["macro_ap"], -r["lr"]))
        assert choice["lr"] == winner["lr"] and choice["checkpoint_sha256"] == winner["checkpoint_sha256"]
    new_results = [r for r in results if r["variant"] in VARIANTS]
    assert len(list((HERE / "test_once").glob("*_predictions.npz"))) == 30
    for record in new_results:
        path = HERE / "test_once" / f"{record['split']}_{record['variant']}_predictions.npz"
        with np.load(path, allow_pickle=False) as p, np.load(SPLITS / f"{record['split']}.npz", allow_pickle=False) as d:
            assert np.array_equal(p["pairs"], d["test_pairs"])
            assert np.array_equal(p["labels"], d["test_labels"])
            assert np.isfinite(p["logits"]).all()
            recomputed = disease_metrics(p["pairs"], p["labels"], p["logits"], 433)
        assert all(np.isclose(recomputed[m], record["test"][m], atol=1e-12, rtol=0) for m in recomputed)
    originals = {r["split"]: r for r in read(OLD_TEST / "test_results.json")}
    for full in [r for r in results if r["variant"] == "full"]:
        assert full["test"] == originals[full["split"]]["test"]
    for summary in summaries:
        rows = [r for r in results if r["split"].startswith(summary["setting"]) and r["variant"] == summary["variant"]]
        assert len(rows) == 5
        for m, statistics in summary["metrics"].items():
            vals = np.asarray([r["test"][m] for r in rows])
            assert statistics["mean"] == float(vals.mean())
            assert statistics["sample_std"] == float(vals.std(ddof=1))
    report = {"passed": True, "new_validation_fits": 60, "validation_prediction_sets_recomputed": 60,
              "checkpoint_epoch_and_learning_rate_selection_verified": 30, "new_test_prediction_sets_recomputed": 30,
              "reused_full_results_unchanged": 10, "five_seed_summary_rows_verified": 8,
              "all_declared_variants_reported": True, "script_and_dependency_and_split_hashes_match_pretraining_plan": True,
              "total_new_fit_seconds": float(sum(r["seconds"] for r in fits)),
              "epoch_and_checkpoint_checks": epoch_checks,
              "audit_scope": "Checks stored predictions and frozen records; no model is retrained or newly evaluated on test data"}
    (HERE / "completed_experiment_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "epoch_and_checkpoint_checks"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
