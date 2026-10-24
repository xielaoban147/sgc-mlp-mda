"""Regression checks for cached-structure recovery and target-free encoding."""
import copy
import csv
import json
from pathlib import Path
import unittest

import numpy as np

from features import _cached_details, _molecules, _name_tokens, _tree_tokens, _HashSection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'outputs/代谢物疾病研究入门/数据与代码/CTD训练候选快照'
CACHE = ROOT / 'work/plan_refresh/dataset_audit/demo_dataset/raw'

class FeatureRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (SOURCE / 'metabolites.tsv').open(encoding='utf-8-sig') as stream:
            cls.rows = {r['chebi_id']: r for r in csv.DictReader(stream, delimiter='\t')}
        cls.details = _cached_details(CACHE, ['CHEBI:15551', 'CHEBI:16235'])

    def test_restore_stereo_only_from_verified_same_id_detail(self):
        features, exclusions, audit, _ = _molecules(self.rows, ['CHEBI:15551'], self.details)
        self.assertEqual(set(features), {'CHEBI:15551'})
        self.assertFalse(exclusions)
        self.assertTrue(audit[0]['cached_detail_used'])
        self.assertNotEqual(audit[0]['source_smiles'], audit[0]['used_smiles'])
        self.assertEqual(audit[0]['recomputed_inchikey'], self.rows['CHEBI:15551']['inchikey'])
        self.assertTrue(audit[0]['cache_body_sha256'])

    def test_reject_detail_when_recomputed_complete_key_disagrees(self):
        features, exclusions, audit, _ = _molecules(self.rows, ['CHEBI:16235'], self.details)
        self.assertFalse(features)
        self.assertEqual(len(exclusions), 1)
        self.assertFalse(audit[0]['cached_detail_used'])

    def test_reject_declared_key_mismatch_even_when_smiles_would_match(self):
        details = copy.deepcopy(self.details)
        for candidate in details['CHEBI:15551']:
            candidate['declared_inchikey'] = 'WRONG-FULL-KEY'
        features, exclusions, _, _ = _molecules(self.rows, ['CHEBI:15551'], details)
        self.assertFalse(features)
        self.assertEqual(len(exclusions), 1)

    def test_multitree_ancestors_and_top_level_zero(self):
        own, tokens = _tree_tokens('C18.452.394.750|C18.452|C10.100')
        self.assertEqual(tokens, {'tree:C18', 'tree:C18.452.394', 'tree:C10'})
        self.assertEqual(_tree_tokens('C08')[1], set())

    def test_normalization_and_stable_binary_hash(self):
        normalized, tokens = _name_tokens('  Ｄiabetes\tMELLITUS  ')
        self.assertEqual(normalized, 'diabetes mellitus')
        self.assertIn('word2:diabetes mellitus', tokens)
        section = _HashSection(512)
        a, b = section.encode(tokens), section.encode(set(reversed(sorted(tokens))))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (512,))
        self.assertTrue(np.isin(a, [0, 1]).all())

    def test_real_final_features_are_aligned_and_finite(self):
        manifest = json.loads((HERE / 'data/features/manifest.json').read_text(encoding='utf-8'))
        with np.load(HERE / 'data/features/dataset.npz', allow_pickle=False) as dataset:
            fm, fd, p = (dataset[k] for k in ('metabolite_features', 'disease_features', 'positive_pairs'))
            self.assertEqual(fm.shape, (234, 2048))
            self.assertEqual(fd.shape, (433, 1536))
            self.assertEqual(p.shape, (3184, 2))
            for matrix in (fm, fd):
                self.assertTrue(np.isfinite(matrix).all())
                self.assertTrue(np.isin(matrix, [0, 1]).all())
            self.assertEqual(len(set(map(tuple, p.tolist()))), len(p))
            self.assertGreaterEqual(np.bincount(p[:, 1], minlength=433).min(), 3)
            self.assertTrue(all(np.array_equal(dataset[k], np.sort(dataset[k])) for k in ('metabolite_ids', 'disease_ids')))
        self.assertEqual(manifest['cached_detail_restored_nodes'], 19)
        self.assertEqual(manifest['structure_excluded_nodes'], 35)

if __name__ == '__main__':
    unittest.main(verbosity=2)
