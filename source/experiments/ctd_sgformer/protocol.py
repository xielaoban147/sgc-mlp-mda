"""Frozen CTD splits, training-only contexts, and disease-level metrics.

Indices in every saved split follow lexicographically sorted normalized IDs.
Use the features saved in that split, or its source-index arrays, to avoid an
ID/feature mismatch when the input dataset was not already sorted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SEEDS = (17, 42, 2026)


def _pairs_array(value: np.ndarray, num_m: int, num_d: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.dtype.kind not in "iu":
        raise ValueError("Pairs must be an integer array with shape (N, 2).")
    arr = arr.astype(np.int64, copy=True)
    if (arr < 0).any() or (arr[:, 0] >= num_m).any() or (arr[:, 1] >= num_d).any():
        raise ValueError("A pair contains an out-of-range node index.")
    return arr


def _sort_pairs(pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].copy()


def _keys(pairs: np.ndarray, num_d: int) -> np.ndarray:
    return pairs[:, 0] * num_d + pairs[:, 1]


def _from_keys(keys: np.ndarray, num_d: int) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    return np.column_stack((keys // num_d, keys % num_d)).astype(np.int64)


def _ids(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.dtype.kind not in "US" or len(arr) == 0:
        raise ValueError(f"{name} must be a nonempty string vector, not an object array.")
    arr = arr.astype(str)
    if (arr == "").any() or len(np.unique(arr)) != len(arr):
        raise ValueError(f"{name} must contain unique, nonempty IDs.")
    return arr


def _load_canonical_dataset(path: Path) -> dict[str, np.ndarray]:
    required = {
        "positive_pairs", "metabolite_ids", "disease_ids",
        "metabolite_features", "disease_features",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Dataset is missing keys: {sorted(missing)}")
        m_ids = _ids(data["metabolite_ids"], "metabolite_ids")
        d_ids = _ids(data["disease_ids"], "disease_ids")
        num_m, num_d = len(m_ids), len(d_ids)
        positives = _pairs_array(data["positive_pairs"], num_m, num_d)
        if len(positives) == 0:
            raise ValueError("Dataset contains no positive pairs.")
        if len(np.unique(_keys(positives, num_d))) != len(positives):
            raise ValueError("Duplicate positive pairs must be resolved before splitting.")
        m_features = np.asarray(data["metabolite_features"])
        d_features = np.asarray(data["disease_features"])
        for features, count, name in (
            (m_features, num_m, "metabolite_features"),
            (d_features, num_d, "disease_features"),
        ):
            if features.ndim != 2 or features.shape[0] != count:
                raise ValueError(f"{name} is not aligned with its IDs.")
            if features.dtype.kind not in "biuf" or not np.isfinite(features).all():
                raise ValueError(f"{name} must contain finite numeric values.")
        m_order = np.argsort(m_ids, kind="stable")
        d_order = np.argsort(d_ids, kind="stable")
        m_inverse = np.empty(num_m, dtype=np.int64)
        d_inverse = np.empty(num_d, dtype=np.int64)
        m_inverse[m_order] = np.arange(num_m)
        d_inverse[d_order] = np.arange(num_d)
        positives[:, 0] = m_inverse[positives[:, 0]]
        positives[:, 1] = d_inverse[positives[:, 1]]
        return {
            "positive_pairs": _sort_pairs(positives),
            "metabolite_ids": m_ids[m_order],
            "disease_ids": d_ids[d_order],
            "metabolite_features": m_features[m_order].copy(),
            "disease_features": d_features[d_order].copy(),
            "metabolite_source_indices": m_order.astype(np.int64),
            "disease_source_indices": d_order.astype(np.int64),
        }


def _labelled_pairs(positives: np.ndarray, unknowns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.concatenate((positives, unknowns), axis=0)
    labels = np.concatenate((np.ones(len(positives), dtype=np.int8),
                             np.zeros(len(unknowns), dtype=np.int8)))
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order].copy(), labels[order].copy()


def _split_arrays(dataset: dict[str, np.ndarray], mode: str, seed: int) -> dict[str, np.ndarray]:
    num_m, num_d = len(dataset["metabolite_ids"]), len(dataset["disease_ids"])
    positives = dataset["positive_pairs"]
    universe = np.arange(num_m * num_d, dtype=np.int64)
    unknown_keys = np.setdiff1d(universe, _keys(positives, num_d), assume_unique=True)
    rng = np.random.default_rng(seed)
    if mode == "random_edge":
        shuffled = positives[rng.permutation(len(positives))]
        n_train, n_val = (8 * len(positives)) // 10, len(positives) // 10
        train_pos = shuffled[:n_train]
        val_pos = shuffled[n_train:n_train + n_val]
        test_pos = shuffled[n_train + n_val:]
        if len(unknown_keys) < len(train_pos):
            raise ValueError("Not enough unknown pairs for 1:1 random-edge training.")
        unknowns = rng.permutation(unknown_keys)
        train_neg = _from_keys(unknowns[:len(train_pos)], num_d)
        remaining = unknowns[len(train_pos):]
        midpoint = len(remaining) // 2
        val_neg = _from_keys(remaining[:midpoint], num_d)
        test_neg = _from_keys(remaining[midpoint:], num_d)
        # Random edge splitting has NO mutually exclusive disease partition.
        train_diseases = val_diseases = test_diseases = np.arange(num_d, dtype=np.int64)
    elif mode == "cold_disease":
        shuffled_diseases = rng.permutation(num_d)
        n_train, n_val = (7 * num_d) // 10, (15 * num_d) // 100
        train_diseases = np.sort(shuffled_diseases[:n_train]).astype(np.int64)
        val_diseases = np.sort(shuffled_diseases[n_train:n_train + n_val]).astype(np.int64)
        test_diseases = np.sort(shuffled_diseases[n_train + n_val:]).astype(np.int64)
        train_pos = positives[np.isin(positives[:, 1], train_diseases)]
        val_pos = positives[np.isin(positives[:, 1], val_diseases)]
        test_pos = positives[np.isin(positives[:, 1], test_diseases)]
        allowed = unknown_keys[np.isin(unknown_keys % num_d, train_diseases)]
        if len(allowed) < len(train_pos):
            raise ValueError("Not enough train-disease unknowns for 1:1 cold-start training.")
        train_neg = _from_keys(rng.permutation(allowed)[:len(train_pos)], num_d)
        val_neg = _from_keys(unknown_keys[np.isin(unknown_keys % num_d, val_diseases)], num_d)
        test_neg = _from_keys(unknown_keys[np.isin(unknown_keys % num_d, test_diseases)], num_d)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    val_pairs, val_labels = _labelled_pairs(val_pos, val_neg)
    test_pairs, test_labels = _labelled_pairs(test_pos, test_neg)
    return {
        **{key: value.copy() for key, value in dataset.items()},
        "train_pos": _sort_pairs(train_pos),
        "train_neg": _sort_pairs(train_neg),
        "val_pairs": val_pairs,
        "val_labels": val_labels,
        "test_pairs": test_pairs,
        "test_labels": test_labels,
        "train_diseases": train_diseases.copy(),
        "val_diseases": val_diseases.copy(),
        "test_diseases": test_diseases.copy(),
        "num_m": np.asarray(num_m, dtype=np.int64),
        "num_d": np.asarray(num_d, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "mode": np.asarray(mode),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_splits(dataset_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Write six deterministic NPZs and companion manifests; return NPZ paths."""
    source = Path(dataset_path).resolve()
    destination = Path(out_dir).resolve()
    dataset = _load_canonical_dataset(source)
    destination.mkdir(parents=True, exist_ok=True)
    source_digest = _sha256(source)
    paths: list[Path] = []
    for mode in ("random_edge", "cold_disease"):
        for seed in SEEDS:
            arrays = _split_arrays(dataset, mode, seed)
            path = destination / f"{mode}_seed{seed}.npz"
            np.savez_compressed(path, **arrays)
            manifest = {
                "protocol_version": 1,
                "mode": mode,
                "seed": seed,
                "source": str(source),
                "source_sha256": source_digest,
                "split_sha256": _sha256(path),
                "numpy_version": np.__version__,
                "index_order": "lexicographically sorted normalized metabolite/disease IDs",
                "random_generator": "numpy.default_rng / PCG64",
                "train_unknown_semantics": "fixed unlabelled pairs treated as 0; not certified negatives",
                "scoring_context": "all train_pos; no query removal, augmentation, or train+val refit",
                "unknown_pool": "exclude every recorded positive; exclude held-out diseases for cold training",
                "num_m": int(arrays["num_m"]),
                "num_d": int(arrays["num_d"]),
                "num_positive_pairs": len(dataset["positive_pairs"]),
                "counts": {
                    "train_pos": len(arrays["train_pos"]),
                    "train_neg": len(arrays["train_neg"]),
                    "val_pos": int(arrays["val_labels"].sum()),
                    "val_unknown": int((arrays["val_labels"] == 0).sum()),
                    "val_pairs": len(arrays["val_pairs"]),
                    "test_pos": int(arrays["test_labels"].sum()),
                    "test_unknown": int((arrays["test_labels"] == 0).sum()),
                    "test_pairs": len(arrays["test_pairs"]),
                    "train_diseases": len(arrays["train_diseases"]),
                    "val_diseases": len(arrays["val_diseases"]),
                    "test_diseases": len(arrays["test_diseases"]),
                },
                "disease_sets_are_disjoint": mode == "cold_disease",
                "cold_start_scope": "transductive; every node's independent attributes remain available",
                "files_contain_canonical_features": True,
            }
            path.with_suffix(".manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            paths.append(path)
    return paths


def _context_graph(pairs: np.ndarray, num_m: int, num_d: int) -> dict[str, np.ndarray]:
    pairs = _sort_pairs(pairs)
    m, d = pairs[:, 0], pairs[:, 1] + num_m
    edge_index = np.stack((np.concatenate((m, d)), np.concatenate((d, m)))).astype(np.int64)
    degree = np.bincount(np.concatenate((m, d)), minlength=num_m + num_d).astype(np.int64)
    return {"pairs": pairs, "edge_index": edge_index, "degree": degree}


def make_context(
    train_pos: np.ndarray,
    batch_pos: np.ndarray,
    num_m: int,
    num_d: int,
    train_diseases: np.ndarray,
    seed: int,
) -> dict:
    """Make all three contexts from the immutable training positives only.

    This function knows no validation/test data. In random_edge, pass all
    disease indices; in cold_disease, pass the frozen training disease set.
    `deleted_count` excludes the removal of the supervised batch queries.
    """
    if num_m <= 0 or num_d <= 0:
        raise ValueError("Node counts must be positive.")
    training = _pairs_array(train_pos, num_m, num_d)
    queries = _pairs_array(batch_pos, num_m, num_d)
    training_keys, query_keys = _keys(training, num_d), _keys(queries, num_d)
    if len(np.unique(training_keys)) != len(training_keys):
        raise ValueError("Training positives must be unique.")
    if not np.isin(query_keys, training_keys).all():
        raise ValueError("Every batch positive must belong to train_pos.")
    allowed = np.asarray(train_diseases)
    if allowed.ndim != 1 or allowed.dtype.kind not in "iu":
        raise ValueError("train_diseases must be an integer vector.")
    allowed = np.unique(allowed.astype(np.int64))
    if (allowed < 0).any() or (allowed >= num_d).any():
        raise ValueError("Out-of-range training disease.")
    if not np.isin(training[:, 1], allowed).all():
        raise ValueError("train_pos contains a disease outside train_diseases.")
    base = _sort_pairs(training[~np.isin(training_keys, query_keys)])
    budget_rng, uniform_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2))
    keep = np.ones(len(base), dtype=bool)
    eligible = np.intersect1d(np.unique(base[:, 1]), allowed)
    if len(eligible):
        count = max(1, (len(eligible) + 9) // 10)
        selected = budget_rng.choice(eligible, size=count, replace=False)
        for disease in selected:
            positions = np.flatnonzero(base[:, 1] == disease)
            choice = int(budget_rng.integers(4))
            retain = (0, 1, 3, len(positions))[choice]
            retain = min(retain, len(positions))
            keep[positions] = False
            if retain:
                keep[budget_rng.choice(positions, size=retain, replace=False)] = True
    removed = int((~keep).sum())
    uniform_keep = np.ones(len(base), dtype=bool)
    if removed:
        uniform_keep[uniform_rng.choice(len(base), size=removed, replace=False)] = False
    return {
        "none": _context_graph(base, num_m, num_d),
        "budget": _context_graph(base[keep], num_m, num_d),
        "uniform": _context_graph(base[uniform_keep], num_m, num_d),
        "deleted_count": removed,
    }


def _metric_inputs(pairs: np.ndarray, labels: np.ndarray, scores: np.ndarray, num_d: int):
    pairs = np.asarray(pairs)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("Metric pairs must be an integer array of shape (N, 2).")
    if num_d <= 0 or (pairs < 0).any() or (pairs[:, 1] >= num_d).any():
        raise ValueError("Invalid disease count or pair indices.")
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(pairs) or len(scores) != len(pairs):
        raise ValueError("Metric labels/scores must align with pairs.")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be 0/1, and scores must be finite.")
    return pairs, labels.astype(np.int8), scores


def _disease_groups(pairs: np.ndarray) -> Iterator[np.ndarray]:
    if len(pairs) == 0:
        return
    order = np.argsort(pairs[:, 1], kind="stable")
    boundary = np.flatnonzero(np.diff(pairs[order, 1])) + 1
    for group in np.split(order, boundary):
        yield group


def _fast_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Unweighted sklearn AP: accumulate precision at whole tied-score groups."""
    order = np.argsort(-scores, kind="stable")
    ranked_labels, ranked_scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(labels) - 1]
    true_positives = np.cumsum(ranked_labels, dtype=np.int64)[ends]
    increments = np.diff(np.r_[0, true_positives])
    precision = true_positives / (ends + 1)
    return float(np.sum(precision * increments) / int(labels.sum()))


def validation_macro_ap(pairs: np.ndarray, labels: np.ndarray, scores: np.ndarray, num_d: int) -> float:
    """Fast validation AP only; eligible diseases use this evaluation's labels."""
    pairs, labels, scores = _metric_inputs(pairs, labels, scores, num_d)
    aps = [_fast_average_precision(labels[group], scores[group])
           for group in _disease_groups(pairs) if labels[group].sum() > 0]
    return float(np.mean(aps)) if aps else float("nan")


def disease_metrics(pairs: np.ndarray, labels: np.ndarray, scores: np.ndarray, num_d: int) -> dict:
    """Macro AP/Recall@20/AUROC, using current-set disease eligibility only."""
    pairs, labels, scores = _metric_inputs(pairs, labels, scores, num_d)
    aps, recalls, aucs = [], [], []
    for group in _disease_groups(pairs):
        y, s = labels[group], scores[group]
        positive_count = int(y.sum())
        if positive_count == 0:
            continue
        aps.append(float(average_precision_score(y, s)))
        ranking = np.lexsort((pairs[group, 0], -s))
        recalls.append(float(y[ranking[:20]].sum() / positive_count))
        if positive_count < len(y):
            aucs.append(float(roc_auc_score(y, s)))
    return {
        "macro_ap": float(np.mean(aps)) if aps else float("nan"),
        "macro_recall20": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_ap_diseases": len(aps),
        "n_auroc_diseases": len(aucs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for path in prepare_splits(args.dataset, args.out):
        print(path)


if __name__ == "__main__":
    main()
