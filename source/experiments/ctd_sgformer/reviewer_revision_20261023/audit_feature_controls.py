"""Check the controlled feature removals independently of test performance."""
import json
from pathlib import Path

from run_decoder_ablation import ALL_VARIANTS, HERE, SPLITS, load_runtime


def main():
    np, torch, runner, Model, device = load_runtime()
    generator = torch.Generator().manual_seed(1947)
    left = torch.randn(11, 64, generator=generator)
    right = torch.randn(11, 64, generator=generator)
    states = {}
    results = []
    for variant in ALL_VARIANTS:
        runner._seed_all(17)
        model = Model(2048, 1536, 234, 433, variant=variant)
        states[variant] = model.state_dict()
        features = model._pair_features(left, right)
        assert features.shape == (11, 256)
        assert torch.equal(features[:, :64], left)
        assert torch.equal(features[:, 64:128], right)
        product_active = variant in ("endpoint_product", "full")
        diff_active = variant in ("endpoint_absdiff", "full")
        assert torch.equal(features[:, 128:192], left * right if product_active else torch.zeros_like(left))
        assert torch.equal(features[:, 192:], torch.abs(left - right) if diff_active else torch.zeros_like(left))
        assert sum(p.numel() for p in model.parameters()) == 484033
        model.decoder(features).sum().backward()
        gradient = model.decoder[0].weight.grad
        if not product_active:
            assert torch.count_nonzero(gradient[:, 128:192]) == 0
        if not diff_active:
            assert torch.count_nonzero(gradient[:, 192:]) == 0
        results.append({"variant": variant, "shape": list(features.shape), "nominal_parameters": 484033,
                        "endpoint_blocks_preserved": True, "product_active": product_active,
                        "difference_active": diff_active, "zero_block_weights_have_zero_gradients": True})
    for variant in ALL_VARIANTS:
        assert states[variant].keys() == states["full"].keys()
        assert all(torch.equal(value, states["full"][key]) for key, value in states[variant].items())
    split_stats = []
    for path in sorted(SPLITS.glob("*_seed*.npz")):
        with np.load(path, allow_pickle=False) as d:
            positives = d["train_pos"].copy()
            num_m, num_d = len(d["metabolite_ids"]), len(d["disease_ids"])
            batch = positives[:128]
            context = runner.make_context(positives, batch, num_m, num_d, d["train_diseases"], 19)["none"]
            key = lambda pairs: pairs[:, 0] * num_d + pairs[:, 1]
            assert not np.isin(key(batch), key(context["pairs"])).any()
            assert len(context["pairs"]) == len(positives) - len(batch)
            all_pos_keys = key(d["positive_pairs"])
            assert not np.isin(key(d["train_neg"]), all_pos_keys).any()
            assert len(positives) == len(d["train_neg"])
            if path.stem.startswith("cold_disease"):
                assert not np.isin(positives[:, 1], d["val_diseases"]).any()
                assert not np.isin(positives[:, 1], d["test_diseases"]).any()
            split_stats.append({"split": path.stem, "train_positives": len(positives), "train_unannotated": len(d["train_neg"]),
                                "train_diseases": len(d["train_diseases"]), "validation_diseases": len(d["val_diseases"]),
                                "test_diseases": len(d["test_diseases"]), "validation_positives": int(d["val_labels"].sum()),
                                "validation_unannotated": int((d["val_labels"] == 0).sum()),
                                "test_positives": int(d["test_labels"].sum()), "test_unannotated": int((d["test_labels"] == 0).sum()),
                                "batch_positive_queries_removed": True, "train_neg_excludes_recorded_positives": True})
    out = {"passed": True, "same_seed_initial_parameter_tensors_identical_for_all_variants": True,
           "feature_control_checks": results, "split_stats": split_stats,
           "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
           "note": "This independent control audit reads split test labels only to report fixed split counts; the training/selection script never indexes test fields."}
    (HERE / "feature_control_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "feature_control_audit", "passed": True, "variants": len(results), "splits": len(split_stats)}))


if __name__ == "__main__":
    main()
