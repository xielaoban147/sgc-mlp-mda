"""Prepare independent molecular and ontology attributes for the frozen CTD cohort.

No network access or model training is performed here. The source association file
is used only to fix the retained cohort and positive pairs, never to build node
attributes. This module requires NumPy and RDKit in the project's environment.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import platform
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "molecule": {
        "fingerprint": "binary Morgan",
        "radius": 2,
        "fp_size": 2048,
        "include_chirality": True,
        "sanitization": "RDKit SanitizeMol SANITIZE_ALL",
        "stereochemistry": "AssignStereochemistry cleanIt=False force=True after sanitization",
        "salt_stripping": False,
        "largest_fragment_only": False,
        "neutralization": False,
        "tautomer_standardization": False,
        "inchi_comparison": "recomputed standard full InChIKey equals source full InChIKey",
        "failed_inchi_recomputation": "record as unverified; keep if graph parses and sanitizes",
        "mismatched_nonempty_inchikey": "exclude and retain both keys in the audit",
        "sanitization_charge_change": "record before/after atom formal charges; no additional neutralization",
    },
    "disease": {
        "source": "independent MEDIC disease_metadata.tsv",
        "name_field": "DiseaseName",
        "tree_field": "TreeNumbers",
        "parent_fields_audit_only": ["ParentIDs", "ParentTreeNumbers"],
        "tree_dimension": 1024,
        "name_dimension": 512,
        "tree_rule": "split |, generate strict dot-delimited prefixes per path, union, subtract every complete own path",
        "name_normalization": "Unicode NFKC then casefold then whitespace collapse and strip",
        "word_regex": r"[^\W_]+ (Unicode letters/digits, underscore excluded)",
        "word_orders": [1, 2],
        "bigram_separator": "one ASCII space",
        "character_lengths": [3, 4, 5],
        "character_source": "whole normalized name including internal spaces; no boundary padding",
        "namespaces": {"tree": "tree:", "word_unigram": "word1:", "word_bigram": "word2:", "char_ngram": "char{n}:"},
        "hash": "UTF-8 SHA256, complete 32-byte digest interpreted unsigned big-endian, modulo section dimension",
        "value": "binary presence; duplicate tokens or bucket collisions set a bit once",
        "normalization_after_hashing": "none; both sections concatenate as raw binary vectors",
        "top_level_no_strict_ancestor": "valid zero tree vector, not missing metadata",
    },
    "cohort": {
        "source_expected": "MeSH descriptor training candidate snapshot",
        "positive_rule": "source label=1 and marker/mechanism in DirectEvidence",
        "min_disease_positive_degree": 3,
        "selection_order": "validate structures, exclude verified key conflicts/invalid structures, deduplicate pairs, recompute disease degree>=3, remove metabolite nodes with no retained edges",
        "id_order": "stripped upper-case CHEBI/MESH IDs sorted lexicographically",
        "attributes_do_not_use": ["association profiles", "positive degree", "PMID", "DirectEvidence", "DiseaseCategories", "metabolite therapeutic roles"],
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
        return list(reader)


def _id(value: str) -> str:
    return value.strip().upper()


def _unique_index(rows: list[dict[str, str]], field: str, name: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = _id(row[field])
        if not identifier:
            raise ValueError(f"{name}: empty {field}")
        if identifier in result:
            raise ValueError(f"{name}: duplicate {field} {identifier}; do not pick an arbitrary record")
        result[identifier] = row
    return result


def _name_tokens(name: str) -> tuple[str, set[str]]:
    normalized = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
    words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    tokens = {"word1:" + word for word in words}
    tokens.update("word2:" + first + " " + second for first, second in zip(words, words[1:]))
    for n in (3, 4, 5):
        tokens.update(f"char{n}:" + normalized[start:start+n] for start in range(len(normalized)-n+1))
    return normalized, tokens


def _tree_tokens(value: str) -> tuple[set[str], set[str]]:
    complete = {tree.strip() for tree in value.split("|") if tree.strip()}
    ancestors: set[str] = set()
    for tree in complete:
        if not re.fullmatch(r"[A-Z]\d{2}(?:\.\d{3})*", tree):
            raise ValueError(f"Nonstandard MeSH descriptor tree path {tree!r}; do not guess a slash/OMIM mapping")
        parts = tree.split(".")
        ancestors.update(".".join(parts[:length]) for length in range(1, len(parts)))
    ancestors.difference_update(complete)
    return complete, {"tree:" + ancestor for ancestor in ancestors}


class _HashSection:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.buckets: dict[int, set[str]] = collections.defaultdict(set)
        self.rows_with_collisions = 0
        self.total_row_collision_tokens = 0

    def encode(self, tokens: set[str]) -> np.ndarray:
        result = np.zeros(self.dimension, dtype=np.float32)
        local: dict[int, set[str]] = collections.defaultdict(set)
        for token in sorted(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            position = int.from_bytes(digest, byteorder="big", signed=False) % self.dimension
            local[position].add(token)
            self.buckets[position].add(token)
            result[position] = 1.0
        collisions = len(tokens) - len(local)
        self.rows_with_collisions += int(collisions > 0)
        self.total_row_collision_tokens += collisions
        return result

    def summary(self) -> dict[str, Any]:
        distinct = sum(len(tokens) for tokens in self.buckets.values())
        collision_buckets = {position: sorted(tokens) for position, tokens in sorted(self.buckets.items()) if len(tokens) > 1}
        return {
            "dimension": self.dimension,
            "distinct_tokens": distinct,
            "occupied_buckets": len(self.buckets),
            "collision_buckets": len(collision_buckets),
            "global_extra_tokens_in_shared_buckets": distinct - len(self.buckets),
            "max_bucket_load": max((len(tokens) for tokens in self.buckets.values()), default=0),
            "rows_with_within_row_collisions": self.rows_with_collisions,
            "within_row_extra_tokens_sum": self.total_row_collision_tokens,
            "bucket_tokens_with_collisions": collision_buckets,
        }


def _same_vectors(matrix: np.ndarray, ids: list[str]) -> dict[str, Any]:
    groups: dict[bytes, list[str]] = collections.defaultdict(list)
    for identifier, vector in zip(ids, matrix):
        groups[np.asarray(vector, dtype=np.uint8).tobytes()].append(identifier)
    duplicates = sorted((sorted(group) for group in groups.values() if len(group) > 1), key=lambda group: group[0])
    return {
        "nodes": len(ids),
        "distinct_vectors": len(groups),
        "duplicate_groups": len(duplicates),
        "nodes_in_duplicate_groups": sum(len(group) for group in duplicates),
        "max_identical_group": max((len(group) for group in groups.values()), default=0),
        "groups": duplicates,
    }


def _cached_details(cache: Path | None, ids: list[str]) -> dict[str, list[dict[str, str]]]:
    """Read verified anonymous ChEBI detail responses, never a search/name match."""
    result: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    if cache is None:
        return result
    allowed = set(ids)
    for metadata_path in sorted(Path(cache).glob('*.json')):
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if '/chebi/backend/api/public/compounds/' not in metadata.get('url', ''):
            continue
        if metadata.get('status') != 200 or metadata.get('anonymous') is not True:
            continue
        body_path = metadata_path.with_suffix('.body')
        if not body_path.exists():
            continue
        body_sha = _sha(body_path)
        if body_sha != metadata.get('sha256'):
            raise ValueError(f'Cached ChEBI detail checksum mismatch: {body_path}')
        payload = json.loads(body_path.read_text(encoding='utf-8'))
        for identifier in sorted(allowed.intersection(payload)):
            record = payload[identifier]
            data = record.get('data', {})
            if (record.get('exists') is not True or record.get('primary_chebi_id') != identifier
                    or data.get('chebi_accession') != identifier):
                continue
            structure = data.get('default_structure') or {}
            smiles = structure.get('smiles') or ''
            key = structure.get('standard_inchi_key') or ''
            if smiles and key:
                result[identifier].append(dict(smiles=smiles, declared_inchikey=key,
                    body_path=str(body_path.resolve()), body_sha256=body_sha,
                    metadata_path=str(metadata_path.resolve()), metadata_sha256=_sha(metadata_path)))
    return result


def _molecules(rows: dict[str, dict[str, str]], ids: list[str], details=None) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    try:
        from rdkit import Chem, DataStructs, rdBase
        from rdkit.Chem import inchi, rdFingerprintGenerator
    except ImportError as error:
        raise RuntimeError("RDKit is required. Run this module with the project .venv prepared by the parent agent; no automatic installation is attempted.") from error
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    features: dict[str, np.ndarray] = {}
    exclusions: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for identifier in ids:
        row = rows[identifier]
        smiles = row["smiles"].strip()
        expected = row["inchikey"].strip().upper()
        if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", expected):
            raise ValueError(f"{identifier}: source full InChIKey missing or malformed; inspect the snapshot rather than supplying an invented key")
        item: dict[str, Any] = {
            "metabolite_id": identifier, "chemical_id": _id(row["chemical_id"]), "name": row["name"],
            "source_smiles": smiles, "used_smiles": smiles, "expected_inchikey": expected, "recomputed_inchikey": "",
            "fragments": 0, "multifragment": False, "status": "", "detail": "",
            "source_status": "", "source_recomputed_inchikey": "", "cached_detail_used": False,
            "cache_body_path": "", "cache_body_sha256": "", "cache_metadata_path": "", "cache_metadata_sha256": "",
            "formal_charges_before_sanitize": "", "formal_charges_after_sanitize": "", "sanitize_changed_charges": False,
        }
        def parse_structure(value):
            mol = Chem.MolFromSmiles(value, sanitize=False)
            if mol is None:
                raise ValueError('RDKit MolFromSmiles returned None')
            before = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
            after = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
            return mol, before, after

        molecule = None
        before, after = [], []
        if not smiles:
            item.update(status='excluded_empty_smiles', detail='Cannot encode an empty molecular graph')
        else:
            try:
                molecule, before, after = parse_structure(smiles)
            except Exception as error:
                item.update(status='excluded_invalid_smiles_or_sanitization', detail=f'{type(error).__name__}: {error}')
            else:
                try:
                    actual = inchi.MolToInchiKey(molecule)
                    item['recomputed_inchikey'] = actual or ''
                    if actual and actual != expected:
                        item.update(status='excluded_recomputed_inchikey_mismatch', detail='Nonempty recomputed full InChIKey disagrees with frozen source key')
                    else:
                        item['status'] = 'verified_recomputed_inchikey' if actual else 'kept_inchi_recomputation_unavailable'
                except Exception as error:
                    item.update(status='kept_inchi_recomputation_unavailable', detail=f'{type(error).__name__}: {error}')

        item['source_status'] = item['status']
        item['source_recomputed_inchikey'] = item['recomputed_inchikey']
        if item['status'].startswith('excluded_') and details:
            candidates = [r for r in details.get(identifier, []) if r['declared_inchikey'] == expected]
            # Do not pick between conflicting detail structures with the same ID.
            if len({r['smiles'] for r in candidates}) == 1:
                candidate = sorted(candidates, key=lambda r: r['body_path'])[0]
                try:
                    recovered, recovery_before, recovery_after = parse_structure(candidate['smiles'])
                    recovery_key = inchi.MolToInchiKey(recovered)
                    if recovery_key == expected:
                        molecule, before, after = recovered, recovery_before, recovery_after
                        item.update(status='verified_cached_detail_inchikey', cached_detail_used=True,
                                    used_smiles=candidate['smiles'], recomputed_inchikey=recovery_key,
                                    cache_body_path=candidate['body_path'], cache_body_sha256=candidate['body_sha256'],
                                    cache_metadata_path=candidate['metadata_path'], cache_metadata_sha256=candidate['metadata_sha256'],
                                    detail='Frozen search SMILES failed; same primary ChEBI ID cached detail passes exact full-key verification')
                except Exception:
                    pass  # Keep the original explicit exclusion; never guess a structure.

        if item['status'].startswith('excluded_'):
            exclusions.append(dict(item))
            audit.append(item)
            continue
        if molecule is None:
            raise RuntimeError(f'{identifier}: retained structure has no parsed molecule')
        if molecule is not None:
            item["formal_charges_before_sanitize"] = json.dumps(before, separators=(",", ":"))
            item["formal_charges_after_sanitize"] = json.dumps(after, separators=(",", ":"))
            item["sanitize_changed_charges"] = before != after
            item["fragments"] = len(Chem.GetMolFrags(molecule))
            item["multifragment"] = item["fragments"] > 1
        vector = np.zeros(2048, dtype=np.float32)
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, vector)
        if vector.shape != (2048,) or not np.isin(vector, [0.0, 1.0]).all():
            raise RuntimeError(f"{identifier}: invalid binary Morgan output; stop rather than silently replacing it with zeros")
        features[identifier] = vector
        audit.append(item)
    software = {
        "rdkit": rdBase.rdkitVersion,
        "inchi_available": bool(getattr(inchi, "INCHI_AVAILABLE", False)),
        "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
    }
    return features, exclusions, audit, software


def prepare_dataset(source_dir: Path, out_dir: Path, chebi_detail_cache: Path | None = None) -> Path:
    """Return dataset.npz after auditable structure checks and degree reselection.

    IDs are CHEBI IDs for metabolites and MESH descriptor IDs for diseases. Pairs
    use local zero-based indices in those separate, lexicographically sorted ID
    arrays. All feature arrays are float32 binary attributes, without normalization.
    """
    source_dir = Path(source_dir).resolve()
    out_dir = Path(out_dir).resolve()
    association_path = source_dir / "association.tsv"
    metabolite_path = source_dir / "metabolites.tsv"
    disease_path = source_dir / "disease_metadata.tsv"
    sources = {name: {"path": str(path), "sha256": _sha(path)} for name, path in
               [("association.tsv", association_path), ("metabolites.tsv", metabolite_path), ("disease_metadata.tsv", disease_path)]}
    associations = _read(association_path, {"chemical_id", "chebi_id", "disease_id", "direct_evidence", "label"})
    molecule_rows = _unique_index(_read(metabolite_path, {"chebi_id", "chemical_id", "name", "inchikey", "smiles"}), "chebi_id", "molecule metadata")
    disease_rows = _unique_index(_read(disease_path, {"DiseaseID", "DiseaseName", "TreeNumbers", "ParentIDs", "ParentTreeNumbers"}), "DiseaseID", "MEDIC metadata")
    chemical_ids = [_id(row["chemical_id"]) for row in molecule_rows.values()]
    if len(set(chemical_ids)) != len(chemical_ids):
        raise ValueError("Multiple CHEBI IDs share one CTD chemical ID in source metadata; do not duplicate their labels")
    original_pairs: set[tuple[str, str]] = set()
    for row in associations:
        metabolite, disease = _id(row["chebi_id"]), _id(row["disease_id"])
        if row["label"].strip() != "1" or "marker/mechanism" not in row["direct_evidence"].split("|"):
            raise ValueError("Input contains a nonpositive or non-marker/mechanism row; inspect the snapshot")
        if metabolite not in molecule_rows or disease not in disease_rows:
            raise ValueError(f"No independent metadata for pair {(metabolite, disease)}")
        if _id(row["chemical_id"]) != _id(molecule_rows[metabolite]["chemical_id"]):
            raise ValueError(f"Source association/metadata identity mismatch for {metabolite}")
        if not disease.startswith("MESH:D"):
            raise ValueError(f"{disease} is not a MeSH descriptor candidate; do not infer an OMIM/supplementary mapping")
        original_pairs.add((metabolite, disease))
    if not original_pairs:
        raise ValueError("Input has no positive pairs")
    source_metabolites = sorted({metabolite for metabolite, _ in original_pairs})
    details = _cached_details(chebi_detail_cache, source_metabolites)
    molecular_features, structure_exclusions, molecule_audit, software = _molecules(molecule_rows, source_metabolites, details)
    after_structure = {(metabolite, disease) for metabolite, disease in original_pairs if metabolite in molecular_features}
    disease_degree = collections.Counter(disease for _, disease in after_structure)
    kept_diseases = {disease for disease, degree in disease_degree.items() if degree >= 3}
    final_pairs = sorted((metabolite, disease) for metabolite, disease in after_structure if disease in kept_diseases)
    metabolite_ids = sorted({metabolite for metabolite, _ in final_pairs})
    disease_ids = sorted({disease for _, disease in final_pairs})
    if not final_pairs:
        raise ValueError("Structure validation and disease degree>=3 retained no training cohort")
    m_index = {identifier: position for position, identifier in enumerate(metabolite_ids)}
    d_index = {identifier: position for position, identifier in enumerate(disease_ids)}
    m_matrix = np.stack([molecular_features[identifier] for identifier in metabolite_ids]).astype(np.float32)
    tree_section, name_section = _HashSection(1024), _HashSection(512)
    disease_vectors, disease_audit = [], []
    for identifier in disease_ids:
        row = disease_rows[identifier]
        trees, tree_tokens = _tree_tokens(row["TreeNumbers"])
        normalized_name, name_tokens = _name_tokens(row["DiseaseName"])
        tree_vector, name_vector = tree_section.encode(tree_tokens), name_section.encode(name_tokens)
        disease_vectors.append(np.concatenate([tree_vector, name_vector]).astype(np.float32))
        disease_audit.append({
            "disease_id": identifier, "name": row["DiseaseName"], "normalized_name": normalized_name,
            "complete_tree_numbers": "|".join(sorted(trees)), "strict_ancestor_tokens": "|".join(sorted(tree_tokens)),
            "parent_ids_audit_only": row["ParentIDs"], "parent_tree_numbers_audit_only": row["ParentTreeNumbers"],
            "tree_missing": not trees, "top_level_no_strict_ancestors": bool(trees) and not tree_tokens,
            "name_missing": not normalized_name, "tree_bits_set": int(tree_vector.sum()),
            "name_bits_set": int(name_vector.sum()), "degree_after_structure_check": disease_degree[identifier],
        })
    d_matrix = np.stack(disease_vectors).astype(np.float32)
    indexed_pairs = np.asarray([(m_index[metabolite], d_index[disease]) for metabolite, disease in final_pairs], dtype=np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.npz"
    np.savez_compressed(dataset_path,
                        metabolite_features=m_matrix, disease_features=d_matrix, positive_pairs=indexed_pairs,
                        metabolite_ids=np.asarray(metabolite_ids, dtype="U"), disease_ids=np.asarray(disease_ids, dtype="U"),
                        metabolite_names=np.asarray([molecule_rows[identifier]["name"] for identifier in metabolite_ids], dtype="U"),
                        disease_names=np.asarray([disease_rows[identifier]["DiseaseName"] for identifier in disease_ids], dtype="U"))
    molecule_fields = list(molecule_audit[0])
    _tsv(out_dir / "molecule_audit.tsv", molecule_audit, molecule_fields)
    _tsv(out_dir / "structure_exclusions.tsv", structure_exclusions, molecule_fields)
    _tsv(out_dir / "disease_attribute_audit.tsv", disease_audit, list(disease_audit[0]))
    source_diseases = sorted({disease for _, disease in original_pairs})
    dropped_diseases = [{"disease_id": identifier, "name": disease_rows[identifier]["DiseaseName"],
                         "positive_degree_after_structure": disease_degree[identifier], "reason": "degree_below_3_after_structure_validation"}
                        for identifier in source_diseases if identifier not in kept_diseases]
    _tsv(out_dir / "disease_exclusions.tsv", dropped_diseases, ["disease_id", "name", "positive_degree_after_structure", "reason"])
    dropped_unused = [{"metabolite_id": identifier, "chemical_id": _id(molecule_rows[identifier]["chemical_id"]),
                       "name": molecule_rows[identifier]["name"], "reason": "no_edges_after_disease_degree_reselection"}
                      for identifier in sorted(set(molecular_features)-set(metabolite_ids))]
    _tsv(out_dir / "metabolites_without_retained_edges.tsv", dropped_unused, ["metabolite_id", "chemical_id", "name", "reason"])
    collision_report = {
        "tree_hash": tree_section.summary(), "name_hash": name_section.summary(),
        "identical_vectors": {"morgan": _same_vectors(m_matrix, metabolite_ids),
                              "tree_only": _same_vectors(d_matrix[:, :1024], disease_ids),
                              "name_only": _same_vectors(d_matrix[:, 1024:], disease_ids),
                              "combined_disease": _same_vectors(d_matrix, disease_ids)},
    }
    _json(out_dir / "hash_and_vector_audit.json", collision_report)
    config = json.loads(json.dumps(CONFIG))
    config['molecule']['cached_detail_fallback'] = {
        'enabled': chebi_detail_cache is not None,
        'rule': 'only source parse/key failure; same primary CHEBI ID; cached declared and recomputed complete key must both equal frozen expected key; reject ambiguous SMILES',
        'network': False, 'source_snapshot_modified': False}
    _json(out_dir / "feature_config.json", config)
    status_counts = dict(collections.Counter(row["status"] for row in molecule_audit))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_dir": str(source_dir), "sources": sources,
        "config_file": "feature_config.json", "software": software, "feature_module_sha256": _sha(Path(__file__)),
        "source_association_rows": len(associations), "source_unique_positive_pairs": len(original_pairs),
        "source_duplicate_pair_rows": len(associations)-len(original_pairs),
        "source_metabolites": len(source_metabolites), "source_diseases": len(source_diseases),
        "structure_audit_status_counts": status_counts, "structure_excluded_nodes": len(structure_exclusions),
        "cached_detail_restored_nodes": sum(bool(row['cached_detail_used']) for row in molecule_audit),
        "cached_detail_cache": str(Path(chebi_detail_cache).resolve()) if chebi_detail_cache else None,
        "used_cached_detail_sources": {row['cache_body_path']: {'body_sha256': row['cache_body_sha256'],
            'metadata_path': row['cache_metadata_path'], 'metadata_sha256': row['cache_metadata_sha256']}
            for row in molecule_audit if row['cached_detail_used']},
        "structure_valid_multifragment_nodes": sum(bool(row["multifragment"]) for row in molecule_audit if not row["status"].startswith("excluded_")),
        "sanitization_changed_formal_charge_nodes": sum(bool(row["sanitize_changed_charges"]) for row in molecule_audit),
        "diseases_removed_after_degree_reselection": len(dropped_diseases), "unused_metabolites_after_disease_reselection": len(dropped_unused),
        "retained_metabolites": len(metabolite_ids), "retained_diseases": len(disease_ids), "retained_positive_pairs": len(final_pairs),
        "positive_record_density": len(final_pairs)/(len(metabolite_ids)*len(disease_ids)),
        "feature_shapes": {"metabolite_features": list(m_matrix.shape), "disease_features": list(d_matrix.shape), "positive_pairs": list(indexed_pairs.shape)},
        "feature_dtypes": {"metabolite_features": str(m_matrix.dtype), "disease_features": str(d_matrix.dtype), "positive_pairs": str(indexed_pairs.dtype)},
        "disease_tree_missing_nodes": sum(row["tree_missing"] for row in disease_audit),
        "disease_top_level_no_strict_ancestor_nodes": sum(row["top_level_no_strict_ancestors"] for row in disease_audit),
        "disease_name_missing_nodes": sum(row["name_missing"] for row in disease_audit),
        "attributes_are_independent_of_target_edges": True, "model_training_performed": False, "network_access": False,
        "artifacts_sha256": {},
    }
    artifact_names = ["dataset.npz", "feature_config.json", "hash_and_vector_audit.json", "molecule_audit.tsv", "structure_exclusions.tsv",
                      "disease_attribute_audit.tsv", "disease_exclusions.tsv", "metabolites_without_retained_edges.tsv"]
    manifest["artifacts_sha256"] = {name: _sha(out_dir/name) for name in artifact_names}
    _json(out_dir / "manifest.json", manifest)
    (out_dir / "README.md").write_text(
        "# CTD离线属性与固定训练数据\n\n"
        f"来源：{source_dir}\n\n"
        f"本次保留{len(metabolite_ids)}个代谢物、{len(disease_ids)}个疾病、{len(final_pairs)}个唯一阳性对。"
        "这里只完成特征编码和数据冻结，没有模型训练结果。\n\n"
        "dataset.npz中的positive_pairs分别使用metabolite_ids和disease_ids的局部零基索引；ID按字典序排列。"
        "代谢物为float32的2048位手性Morgan二值指纹，疾病为1024位严格祖先哈希加512位名称哈希。"
        "哈希特征和指纹均不读取目标关联、PMID或治疗角色，所有模型应使用同一NPZ。\n\n"
        "结构使用RDKit默认sanitization，不额外中和、移除盐、取最大碎片或执行互变异构体标准化；"
        "默认sanitization如改动原子形式电荷，会在molecule_audit.tsv明确记录。"
        "搜索结构失败时，仅采用已缓存、同一主ChEBI ID、声明和重算完整InChIKey都精确匹配的详情结构，原字段与所用字段、缓存SHA全部留档；不联网补换实体。"
        "非空重算InChIKey不一致条目排除；计算不出InChIKey只记录未完成复核，不将计算失败冒充结构不一致。"
        "树号只有一段的顶层疾病可具有零祖先向量，这是有效层级信息，不意味着本体缺失。\n\n"
        "feature_config.json固定编码规则；manifest.json保存源文件/产物SHA256、版本及筛选损失。"
        "structure_exclusions.tsv与disease_exclusions.tsv记录排除；hash_and_vector_audit.json记录碰撞和相同向量。"
        "CTD记录为公开文献关联证据，未记录对不是已验证阴性；本数据仍包含部分外源代谢物与跨物种证据。\n",
        encoding="utf-8")
    return dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory with the three frozen candidate TSV files")
    parser.add_argument("--out", type=Path, required=True, help="Feature artifact directory")
    parser.add_argument('--chebi-detail-cache', type=Path, help='Previously frozen anonymous detail response cache; exact full-key fallback only')
    args = parser.parse_args()
    dataset = prepare_dataset(args.source, args.out, args.chebi_detail_cache)
    manifest = json.loads((dataset.parent/"manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"dataset": str(dataset), "metabolites": manifest["retained_metabolites"],
                      "diseases": manifest["retained_diseases"], "positive_pairs": manifest["retained_positive_pairs"],
                      "structure_status": manifest["structure_audit_status_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
