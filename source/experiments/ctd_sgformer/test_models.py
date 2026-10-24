"""Small implementation checks; these are not research-result experiments."""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .models import (
        METHODS, SGFormerCore, TransConvLayer, create_model,
        normalized_graph_aggregate,
    )
except ImportError:
    from models import (
        METHODS, SGFormerCore, TransConvLayer, create_model,
        normalized_graph_aggregate,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestModelAdaptation(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.num_m, self.num_d = 4, 4
        self.fm = torch.randn(self.num_m, 2048)
        self.fd = torch.randn(self.num_d, 1536)
        # Disease 7 intentionally has zero target-relation degree.
        pairs = torch.tensor([[0, 1, 1, 2, 3], [4, 4, 5, 6, 6]])
        self.edges = torch.cat([pairs, pairs.flip(0)], dim=1)
        self.degree = torch.bincount(self.edges[1], minlength=8).float()

    def _forward_loss(self, model: nn.Module) -> torch.Tensor:
        z = model(self.fm, self.fd, self.edges, self.degree)
        self.assertEqual(tuple(z.shape), (8, 64))
        self.assertTrue(torch.isfinite(z).all().item())
        # Use a shared scaled-dot link decoder, including the zero-degree disease.
        src = torch.tensor([0, 1, 2, 3, 0, 2])
        dst = torch.tensor([4, 5, 6, 7, 7, 4])
        labels = torch.tensor([1., 1., 1., 0., 0., 0.])
        logits = (z[src] * z[dst]).sum(-1) / math.sqrt(64)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def test_all_methods_finite_forward_backward_and_no_unused_parameters(self) -> None:
        for method in sorted(METHODS):
            with self.subTest(method=method):
                model = create_model(method, 2048, 1536, 4, 4, dropout=0.0)
                loss = self._forward_loss(model)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, f"unused parameter: {method}:{name}")
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                embeddings = [m for m in model.modules() if isinstance(m, nn.Embedding)]
                self.assertEqual(len(embeddings), 1)
                self.assertEqual(embeddings[0].num_embeddings, 2)

    def test_parameter_counts(self) -> None:
        core = SGFormerCore(128, 64, layers=1, dropout=0.0)
        global_core = SGFormerCore(128, 64, layers=1, dropout=0.0, use_graph=False)
        full = create_model("full", 2048, 1536, 4, 4, dropout=0.0)
        self.assertEqual(parameter_count(core), 37824)
        self.assertEqual(parameter_count(global_core), 25152)
        self.assertEqual(parameter_count(full.adapter), 459264)
        self.assertEqual(parameter_count(full.gate), 41)
        self.assertEqual(parameter_count(full), 497129)
        self.assertEqual(parameter_count(create_model("global_only", 2048, 1536, 4, 4)), 484416)

    def test_zero_degree_gate_keeps_the_complete_global_projection(self) -> None:
        model = create_model("full", 2048, 1536, 4, 4, dropout=0.0).eval()
        z = model(self.fm, self.fd, self.edges, self.degree)
        x = model.adapter(self.fm, self.fd)
        expected = model.encoder.fc(model.encoder.trans_conv(x))
        self.assertEqual(model.last_gate[7].item(), 0.0)
        torch.testing.assert_close(z[7], expected[7])
        self.assertGreater(z[7].abs().sum().item(), 0.0)
        self.assertTrue(((model.last_gate >= 0) & (model.last_gate <= 1)).all().item())

    def test_global_only_does_not_register_local_or_gate_modules(self) -> None:
        model = create_model("global_only", 2048, 1536, 4, 4).eval()
        self.assertFalse(hasattr(model.encoder, "graph_conv"))
        self.assertFalse(hasattr(model, "gate"))
        self.assertFalse(any("graph_conv" in n or "gate" in n for n, _ in model.named_parameters()))
        z = model(self.fm, self.fd, self.edges, self.degree)
        empty = torch.empty((2, 0), dtype=torch.long)
        z_without_edges = model(self.fm, self.fd, empty, torch.zeros(8))
        torch.testing.assert_close(z, z_without_edges)

    def test_index_add_equals_author_dense_normalization_including_duplicates(self) -> None:
        # Duplicates and isolated node ensure accumulation and zero-degree handling.
        edges = torch.tensor([[0, 1, 0, 1, 1, 2], [1, 0, 1, 0, 2, 1]])
        x = torch.randn(4, 5, requires_grad=True)
        src, dst = edges
        degree = torch.bincount(dst, minlength=4).float()
        weights = torch.nan_to_num((1.0 / degree[src]).sqrt() *
                                   (1.0 / degree[dst]).sqrt(),
                                   nan=0.0, posinf=0.0, neginf=0.0)
        dense = torch.zeros(4, 4)
        dense.index_put_((dst, src), weights, accumulate=True)
        actual = normalized_graph_aggregate(x, edges)
        torch.testing.assert_close(actual, dense @ x)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertEqual(actual[3].abs().sum().item(), 0.0)

    def test_norm_guard_and_empty_context_are_finite(self) -> None:
        attention = TransConvLayer(64)
        with torch.no_grad():
            attention.Wq.weight.zero_()
            attention.Wq.bias.zero_()
            attention.Wk.weight.zero_()
            attention.Wk.bias.zero_()
        x = torch.randn(8, 64, requires_grad=True)
        output = attention(x)
        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(output, attention.Wv(x))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        model = create_model("full", 2048, 1536, 4, 4, dropout=0.0)
        z = model(self.fm, self.fd, torch.empty((2, 0), dtype=torch.long), torch.zeros(8))
        self.assertTrue(torch.isfinite(z).all().item())
        self.assertTrue((model.last_gate == 0).all().item())

    def test_degree_rule_and_fixed_weights(self) -> None:
        degree_model = create_model("degree_budget", 2048, 1536, 4, 4, c=2.0)
        degree_model(self.fm, self.fd, self.edges, self.degree)
        torch.testing.assert_close(degree_model.last_gate[:, 0], self.degree / (self.degree + 2.0))
        for method in ("sgformer", "fixed_budget", "fixed_uniform"):
            model = create_model(method, 2048, 1536, 4, 4, alpha=0.25)
            model(self.fm, self.fd, self.edges, self.degree)
            torch.testing.assert_close(model.last_gate, torch.full((8, 1), 0.25))

    def test_sgc_changes_with_context_and_never_caches(self) -> None:
        model = create_model("sgc", 2048, 1536, 4, 4).eval()
        self.assertFalse(model.encoder.cached)
        first = model(self.fm, self.fd, self.edges, self.degree)
        second = model(self.fm, self.fd, torch.empty((2, 0), dtype=torch.long), torch.zeros(8))
        self.assertFalse(torch.allclose(first, second))
        self.assertIsNone(model.encoder._cached_x)

    def test_reset_resets_the_output_projection(self) -> None:
        model = create_model("full", 2048, 1536, 4, 4)
        with torch.no_grad():
            model.encoder.fc.weight.fill_(123.0)
        model.reset_parameters()
        self.assertFalse((model.encoder.fc.weight == 123.0).all().item())

    def test_shapes_validation_and_nondefault_depth(self) -> None:
        for method in sorted(METHODS):
            with self.subTest(method=method):
                model = create_model(method, 2048, 1536, 4, 4,
                                     hidden=128, layers=2, dropout=0.0)
                loss = self._forward_loss(model)
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
        model = create_model("full", 2048, 1536, 4, 4)
        with self.assertRaises(ValueError):
            model(self.fm, self.fd, self.edges, torch.ones(7))
        with self.assertRaises(ValueError):
            model(self.fm, self.fd, self.edges, -torch.ones(8))
        with self.assertRaises(ValueError):
            create_model("lstm", 2048, 1536, 4, 4)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
