"""Meaningful protocol checks; no fabricated model performance is produced."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from protocol import disease_metrics, make_context, prepare_splits, validation_macro_ap


def pair_set(pairs):
    return {tuple(row) for row in pairs.tolist()}


class FrozenSplitsTest(unittest.TestCase):
    def test_counts_disjoint_pools_and_cold_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(91)
            keys = rng.choice(270 * 460, size=3621, replace=False)
            positives = np.column_stack((keys // 460, keys % 460)).astype(np.int64)
            dataset = root / "dataset.npz"
            np.savez(dataset, positive_pairs=positives,
                     metabolite_ids=np.array([f"m{i:04d}" for i in range(270)]),
                     disease_ids=np.array([f"d{i:04d}" for i in range(460)]),
                     metabolite_features=np.zeros((270, 2), dtype=np.float32),
                     disease_features=np.zeros((460, 3), dtype=np.float32))
            paths = prepare_splits(dataset, root / "splits")
            self.assertEqual(len(paths), 6)
            reference = {}
            all_positive = pair_set(positives)
            for path in paths:
                with np.load(path, allow_pickle=False) as split:
                    reference[path.name] = {key: split[key].copy() for key in split.files}
                    train = pair_set(split["train_pos"]) | pair_set(split["train_neg"])
                    validation = pair_set(split["val_pairs"])
                    test = pair_set(split["test_pairs"])
                    self.assertFalse(train & validation)
                    self.assertFalse(train & test)
                    self.assertFalse(validation & test)
                    self.assertFalse(pair_set(split["train_neg"]) & all_positive)
                    self.assertFalse(pair_set(split["val_pairs"][split["val_labels"] == 0]) & all_positive)
                    self.assertFalse(pair_set(split["test_pairs"][split["test_labels"] == 0]) & all_positive)
                    self.assertEqual(len(split["train_pos"]), len(split["train_neg"]))
                    if str(split["mode"]) == "random_edge":
                        self.assertEqual(len(split["train_pos"]), 2896)
                        self.assertEqual(int(split["val_labels"].sum()), 362)
                        self.assertEqual(int(split["test_labels"].sum()), 363)
                        self.assertEqual(len(split["val_pairs"]), 59203)
                        self.assertEqual(len(split["test_pairs"]), 59205)
                        self.assertEqual(len(train | validation | test), 270 * 460)
                    else:
                        td, vd, ed = (set(split[key].tolist()) for key in
                                      ("train_diseases", "val_diseases", "test_diseases"))
                        self.assertEqual((len(td), len(vd), len(ed)), (322, 69, 69))
                        self.assertFalse(td & vd or td & ed or vd & ed)
                        self.assertTrue(set(split["train_pos"][:, 1]).issubset(td))
                        self.assertTrue(set(split["train_neg"][:, 1]).issubset(td))
                        self.assertEqual(set(split["val_pairs"][:, 1]), vd)
                        self.assertEqual(set(split["test_pairs"][:, 1]), ed)
                        self.assertEqual(len(validation), 18630)
                        self.assertEqual(len(test), 18630)
                manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["counts"]["train_pos"], len(reference[path.name]["train_pos"]))
                self.assertEqual(len(manifest["source_sha256"]), 64)
            # Array equality, not ZIP timestamps, is the reproducibility boundary.
            for path in prepare_splits(dataset, root / "repeat"):
                with np.load(path, allow_pickle=False) as repeated:
                    for key, expected in reference[path.name].items():
                        np.testing.assert_array_equal(repeated[key], expected)

    def test_unsorted_ids_realign_features_and_positive_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dataset.npz"
            original = np.array([[0, 0], [1, 2], [2, 1], [0, 2]], dtype=np.int64)
            np.savez(source, positive_pairs=original,
                     metabolite_ids=np.array(["m9", "m1", "m3"]),
                     disease_ids=np.array(["d3", "d1", "d2"]),
                     metabolite_features=np.array([[90], [10], [30]], dtype=np.float32),
                     disease_features=np.array([[30], [10], [20]], dtype=np.float32))
            with np.load(prepare_splits(source, root / "splits")[0], allow_pickle=False) as split:
                np.testing.assert_array_equal(split["metabolite_ids"], ["m1", "m3", "m9"])
                np.testing.assert_array_equal(split["disease_ids"], ["d1", "d2", "d3"])
                np.testing.assert_array_equal(split["metabolite_features"].ravel(), [10, 30, 90])
                np.testing.assert_array_equal(split["disease_features"].ravel(), [10, 20, 30])
                self.assertEqual(pair_set(split["positive_pairs"]), {(2, 2), (0, 1), (1, 0), (2, 1)})


class ContextTest(unittest.TestCase):
    def test_immutable_query_removal_bidirectionality_and_matched_deletion(self):
        training = np.array([(m, d) for m in range(5) for d in range(10)], dtype=np.int64)
        snapshot = training.copy()
        first = np.array([[0, 0], [2, 3]], dtype=np.int64)
        found_nonzero = False
        for seed in range(30):
            context = make_context(training, first, 5, 12, np.arange(10), seed)
            np.testing.assert_array_equal(training, snapshot)
            base_set = pair_set(context["none"]["pairs"])
            self.assertEqual(base_set, pair_set(training) - pair_set(first))
            removed = context["deleted_count"]
            found_nonzero |= removed > 0
            self.assertEqual(len(context["budget"]["pairs"]), len(base_set) - removed)
            self.assertEqual(len(context["uniform"]["pairs"]), len(base_set) - removed)
            for variant in ("none", "budget", "uniform"):
                graph = context[variant]
                self.assertTrue(pair_set(graph["pairs"]).issubset(base_set))
                edge_set = pair_set(graph["edge_index"].T)
                for metabolite, disease in graph["pairs"]:
                    self.assertIn((int(metabolite), int(disease + 5)), edge_set)
                    self.assertIn((int(disease + 5), int(metabolite)), edge_set)
                self.assertEqual(len(edge_set), 2 * len(graph["pairs"]))
                expected = np.bincount(graph["edge_index"][0], minlength=17)
                np.testing.assert_array_equal(graph["degree"], expected)
                np.testing.assert_array_equal(graph["degree"][-2:], [0, 0])
                self.assertEqual(int(graph["degree"].sum()), 2 * len(graph["pairs"]))
            repeat = make_context(training, first, 5, 12, np.arange(10), seed)
            for variant in ("none", "budget", "uniform"):
                for key in ("pairs", "edge_index", "degree"):
                    np.testing.assert_array_equal(context[variant][key], repeat[variant][key])
        self.assertTrue(found_nonzero)
        # A previous batch's query is restored when it is not in the new batch.
        next_context = make_context(training, np.array([[4, 9]]), 5, 12, np.arange(10), 11)
        self.assertTrue(pair_set(first).issubset(pair_set(next_context["none"]["pairs"])))

    def test_reject_nontraining_query_and_heldout_disease(self):
        training = np.array([[0, 0], [1, 0]], dtype=np.int64)
        with self.assertRaises(ValueError):
            make_context(training, np.array([[0, 1]]), 2, 2, np.array([0]), 17)
        with self.assertRaises(ValueError):
            make_context(training, np.empty((0, 2), dtype=np.int64), 2, 2, np.array([1]), 17)
        empty = make_context(training, training, 2, 2, np.array([0]), 17)
        self.assertEqual(empty["deleted_count"], 0)
        for variant in ("none", "budget", "uniform"):
            self.assertEqual(empty[variant]["edge_index"].shape, (2, 0))
            np.testing.assert_array_equal(empty[variant]["degree"], np.zeros(4, dtype=np.int64))


class DiseaseMetricsTest(unittest.TestCase):
    def test_ties_macro_eligibility_and_fixed_id_recall(self):
        pairs = np.array([(m, 0) for m in range(25)] + [(0, 1), (1, 1), (0, 2), (1, 2)])
        labels = np.array([1] + [0] * 23 + [1] + [1, 1, 0, 0], dtype=np.int8)
        scores = np.ones(len(pairs))
        result = disease_metrics(pairs, labels, scores, 4)
        # Disease 0: tied AP = 2/25, fixed-ID top20 recovers 1/2.
        # Disease 1: positive-only AP/recall=1, AUC ineligible.
        # Disease 2: negative-only, all metrics ineligible; disease 3 absent.
        self.assertAlmostEqual(result["macro_ap"], (2 / 25 + 1) / 2)
        self.assertAlmostEqual(result["macro_recall20"], (0.5 + 1) / 2)
        self.assertAlmostEqual(result["macro_auroc"], 0.5)
        self.assertEqual((result["n_ap_diseases"], result["n_auroc_diseases"]), (2, 1))
        self.assertAlmostEqual(validation_macro_ap(pairs, labels, scores, 4), result["macro_ap"])
        reverse = np.arange(len(pairs) - 1, -1, -1)
        self.assertEqual(disease_metrics(pairs[reverse], labels[reverse], scores[reverse], 4), result)

    def test_fast_validation_ap_matches_sklearn_for_tied_scores(self):
        rng = np.random.default_rng(99)
        pairs = np.array([(m, d) for m in range(57) for d in range(7)], dtype=np.int64)
        for _ in range(12):
            labels = rng.integers(0, 2, len(pairs), dtype=np.int8)
            scores = rng.integers(-3, 4, len(pairs)).astype(np.float64)
            expected = np.mean([average_precision_score(labels[pairs[:, 1] == d],
                                                        scores[pairs[:, 1] == d])
                                for d in range(7) if labels[pairs[:, 1] == d].sum()])
            self.assertAlmostEqual(validation_macro_ap(pairs, labels, scores, 7), expected, places=14)
        result = disease_metrics(np.empty((0, 2), dtype=np.int64), np.array([]), np.array([]), 2)
        self.assertTrue(np.isnan(result["macro_ap"]))
        self.assertEqual(result["n_ap_diseases"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
