"""Independent CPU mechanism and protocol checks, not performance experiments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from models_v2 import make_model
from protocol import _load_canonical_dataset, _split_arrays, make_context
import run_revision as revision


def pairs(value):
    return set(map(tuple, value.tolist()))


class RevisionMechanismTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(991)
        self.fm, self.fd = torch.randn(4, 7), torch.randn(4, 9)
        undirected = torch.tensor([[0, 1, 1, 2, 3], [4, 4, 5, 6, 6]])
        self.edges = torch.cat((undirected, undirected.flip(0)), dim=1)
        self.degree = torch.bincount(self.edges[1], minlength=8).float()

    def model(self, method):
        return make_model(method, revision.configs(method)[0], self.fm, self.fd).eval()

    def test_hard_soft_same_state_agree_except_zero_degree_route(self):
        hard, soft = self.model('hard_budget'), self.model('soft_budget')
        soft.load_state_dict(hard.state_dict())
        a = hard(self.fm, self.fd, self.edges, self.degree)
        b = soft(self.fm, self.fd, self.edges, self.degree)
        torch.testing.assert_close(a[self.degree > 0], b[self.degree > 0])
        self.assertTrue(torch.all(hard.last_gate[self.degree == 0] == 0))
        self.assertTrue(torch.all(soft.last_gate[self.degree == 0] > 0))
        self.assertFalse(torch.allclose(a[self.degree == 0], b[self.degree == 0]))

    def test_residual_keeps_attribute_path_without_any_edges(self):
        model = self.model('residual_budget')
        empty, zero = torch.empty((2, 0), dtype=torch.long), torch.zeros(8)
        actual = model(self.fm, self.fd, empty, zero)
        attributes = model.adapter(self.fm, self.fd)
        local = model.encoder.graph_conv
        residual = F.relu(local.bns[0](local.fcs[0](attributes)))
        global_features = model.encoder.trans_conv(attributes)
        expected = model.encoder.fc(0.5 * global_features + 0.5 * residual)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.all(model.last_gate == 0))
        actual.square().sum().backward()
        self.assertGreater(local.fcs[0].weight.grad.abs().sum().item(), 0)

    def test_residual_fixed_matches_fixed_branch_at_positive_degree(self):
        fixed, residual = self.model('fixed_budget'), self.model('residual_fixed_budget')
        residual.load_state_dict(fixed.state_dict())
        a = fixed(self.fm, self.fd, self.edges, self.degree)
        b = residual(self.fm, self.fd, self.edges, self.degree)
        torch.testing.assert_close(a[self.degree > 0], b[self.degree > 0])
        torch.testing.assert_close(residual.last_gate[:, 0], (self.degree > 0).float())
        self.assertFalse(any(name.startswith('gate.') for name in residual.state_dict()))

    def test_all_methods_have_finite_forward_backward(self):
        for method in revision.METHODS:
            with self.subTest(method=method):
                model = self.model(method).train()
                z = model(self.fm, self.fd, self.edges, self.degree)
                self.assertEqual(tuple(z.shape), (8, 64))
                source, target = torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])
                logits = (z[source] * z[target]).sum(-1) / math.sqrt(64)
                loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 1., 0., 0.]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, value in model.named_parameters():
                    self.assertIsNotNone(value.grad, name)
                    self.assertTrue(torch.isfinite(value.grad).all(), name)

    def test_every_variant_can_reset(self):
        for method in ('soft_budget', 'residual_budget', 'residual_none', 'residual_fixed_budget'):
            with self.subTest(method=method):
                model = self.model(method)
                model.reset_parameters()
                self.assertTrue(torch.isfinite(model(self.fm, self.fd, self.edges, self.degree)).all())


class RevisionProtocolTest(unittest.TestCase):
    def test_shared_learning_rates_and_explicit_tuned_budget(self):
        self.assertEqual(len(revision.METHODS), len(set(revision.METHODS)))
        self.assertEqual(sum(len(revision.configs(m)) for m in revision.METHODS), 34)
        for method in revision.METHODS:
            configs = revision.configs(method)
            self.assertEqual({c['lr'] for c in configs}, {0.001, 0.0003})
            self.assertEqual({c['alpha'] for c in configs},
                             {0.2, 0.5, 0.8} if method == 'fixed_tuned' else {0.5})
            self.assertEqual(len(configs), 6 if method == 'fixed_tuned' else 2)
        self.assertEqual(revision.MASK['hard_none'], 'none')
        for method in ('hard_budget', 'soft_budget', 'residual_budget', 'residual_fixed_budget'):
            self.assertEqual(revision.MASK[method], 'budget')
        self.assertEqual(revision.MASK['hard_uniform'], 'uniform')

    def test_original_source_and_data_hashes_preserved(self):
        original = json.loads((BASE / 'runs/20261014_v1/frozen_plan.json').read_text(encoding='utf-8'))
        for name, expected in original['source_sha256'].items():
            self.assertEqual(hashlib.sha256((BASE / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(hashlib.sha256((BASE / 'data/features/dataset.npz').read_bytes()).hexdigest(),
                         original['dataset_sha256'])

    def test_all_ten_split_isolations_and_existing_split_identity(self):
        dataset = _load_canonical_dataset(BASE / 'data/features/dataset.npz')
        all_positive = pairs(dataset['positive_pairs'])
        for mode in ('random_edge', 'cold_disease'):
            for seed in revision.SEEDS:
                with self.subTest(mode=mode, seed=seed):
                    split = _split_arrays(dataset, mode, seed)
                    train = pairs(split['train_pos']) | pairs(split['train_neg'])
                    val, test = pairs(split['val_pairs']), pairs(split['test_pairs'])
                    self.assertFalse(train & val or train & test or val & test)
                    self.assertFalse(pairs(split['train_neg']) & all_positive)
                    for part in ('val', 'test'):
                        self.assertFalse(pairs(split[f'{part}_pairs'][split[f'{part}_labels'] == 0]) & all_positive)
                    if mode == 'cold_disease':
                        td, vd, ed = (set(split[f'{part}_diseases'].tolist()) for part in ('train', 'val', 'test'))
                        self.assertFalse(td & vd or td & ed or vd & ed)
                        self.assertTrue(set(split['train_pos'][:, 1]).issubset(td))
                        self.assertTrue(set(split['train_neg'][:, 1]).issubset(td))
                        self.assertEqual(set(split['val_pairs'][:, 1]), vd)
                        self.assertEqual(set(split['test_pairs'][:, 1]), ed)
                    name = f'{mode}_seed{seed}.npz'
                    for existing in (HERE / 'splits' / name, BASE / 'data/splits' / name):
                        if existing.exists():
                            with np.load(existing, allow_pickle=False) as saved:
                                self.assertEqual(set(saved.files), set(split))
                                for key in saved.files:
                                    np.testing.assert_array_equal(saved[key], split[key])
                    current = make_context(split['train_pos'], split['train_pos'][:128],
                                           int(split['num_m']), int(split['num_d']), split['train_diseases'], 101)
                    base = pairs(current['none']['pairs'])
                    self.assertEqual(base, pairs(split['train_pos']) - pairs(split['train_pos'][:128]))
                    for kind in ('budget', 'uniform'):
                        self.assertEqual(len(base) - len(current[kind]['pairs']), current['deleted_count'])
                        np.testing.assert_array_equal(current[kind]['degree'],
                            np.bincount(current[kind]['edge_index'][1], minlength=int(split['num_m'] + split['num_d'])))
                        self.assertFalse(pairs(current[kind]['pairs']) & (val | test))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
