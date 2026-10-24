"""Lightweight ego/neighbor-separated link predictor.

Design inspiration: Zhu et al., NeurIPS 2024, which studies feature
heterophily in link prediction and motivates separating ego/neighbor signals
and using learnable pair decoders. This is a small task-specific adaptation,
not a reproduction of their full benchmark or models.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models import TypeFeatureAdapter, TransConv, create_model


class EgoNeighborLinkPredictor(nn.Module):
    """Two-branch SAGE-style encoder with a compact pairwise decoder.

    The feature projection for a node (ego) is never blended with its
    association-neighbor aggregate. Pair features combine endpoint states,
    elementwise interactions, and absolute differences. Isolated nodes have
    an exactly zero neighbor state.
    """

    VARIANTS = {"sgc_mlp", "sgformer_mlp", "separated_mlp", "ego_only_mlp", "mixed_mlp", "separated_distmult",
                "global_separated_mlp", "global_mixed_mlp", "global_separated_distmult"}

    def __init__(self, metabolite_dim: int, disease_dim: int, num_m: int,
                 num_d: int, hidden: int = 32, dropout: float = 0.2,
                 variant: str = "separated_mlp"):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.num_m = num_m
        self.num_d = num_d
        self.dropout = dropout
        self.adapter = (None if variant == "sgformer_mlp" else
                        TypeFeatureAdapter(metabolite_dim, disease_dim, num_m, num_d))
        if variant == "sgformer_mlp":
            self.graph_encoder = create_model("sgformer", metabolite_dim, disease_dim,
                                              num_m, num_d, hidden=64, layers=1,
                                              dropout=dropout, alpha=0.5, c=1.0)
            self.decoder = nn.Sequential(
                nn.Linear(4 * 64, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            return
        if variant == "sgc_mlp":
            from torch_geometric.nn import SGConv
            self.graph_encoder = SGConv(128, 64, K=1, cached=False,
                                        add_self_loops=True, bias=True)
            self.decoder = nn.Sequential(
                nn.Linear(4 * 64, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            return
        self.ego_proj = nn.Linear(128, hidden)
        self.use_global = variant.startswith("global_")
        if self.use_global:
            self.global_encoder = TransConv(128, hidden, dropout)
        if variant not in {"ego_only_mlp"}:
            self.neighbor_proj = nn.Linear(128, hidden, bias=False)
        if variant in {"separated_mlp", "ego_only_mlp", "mixed_mlp"}:
            multiplier = 8 if variant == "separated_mlp" else 4
            self.decoder = nn.Sequential(
                nn.Linear(multiplier * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        elif variant == "global_separated_mlp":
            self.decoder = nn.Sequential(
                nn.Linear(12 * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        elif variant == "global_mixed_mlp":
            self.decoder = nn.Sequential(
                nn.Linear(4 * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.ego_weight = nn.Parameter(torch.empty(hidden))
            self.neighbor_weight = nn.Parameter(torch.empty(hidden))
            if variant == "global_separated_distmult":
                self.global_weight = nn.Parameter(torch.empty(hidden))
                nn.init.normal_(self.global_weight, std=hidden ** -0.5)
            self.decoder_bias = nn.Parameter(torch.zeros(()))
            nn.init.normal_(self.ego_weight, std=hidden ** -0.5)
            nn.init.normal_(self.neighbor_weight, std=hidden ** -0.5)

    @staticmethod
    def _mean_neighbor_features(x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        n = x.shape[0]
        src, dst = edge_index
        sums = torch.zeros_like(x).index_add(0, dst, x[src])
        degree = torch.bincount(dst, minlength=n).to(dtype=x.dtype)
        aggregate = sums / degree.clamp_min(1).unsqueeze(-1)
        aggregate = aggregate * (degree > 0).to(x.dtype).unsqueeze(-1)
        return aggregate, degree

    @staticmethod
    def _pair_features(left: Tensor, right: Tensor) -> Tensor:
        return torch.cat((left, right, left * right, torch.abs(left - right)), dim=-1)

    def encode(self, metabolite_features: Tensor, disease_features: Tensor,
               edge_index: Tensor) -> tuple[Tensor, ...]:
        if self.variant == "sgformer_mlp":
            degree = torch.bincount(edge_index[1], minlength=self.num_m + self.num_d).to(torch.float32)
            return self.graph_encoder(metabolite_features, disease_features,
                                      edge_index, degree), None
        assert self.adapter is not None
        x = self.adapter(metabolite_features, disease_features)
        if self.variant in {"sgc_mlp", "sgformer_mlp"}:
            return self.graph_encoder(x, edge_index), None
        ego = F.dropout(F.relu(self.ego_proj(x)), p=self.dropout, training=self.training)
        global_state = self.global_encoder(x) if self.use_global else None
        if self.variant == "ego_only_mlp":
            return ego, None
        neighbor_x, degree = self._mean_neighbor_features(x, edge_index)
        neighbor = F.relu(self.neighbor_proj(neighbor_x))
        neighbor = neighbor * (degree > 0).to(neighbor.dtype).unsqueeze(-1)
        neighbor = F.dropout(neighbor, p=self.dropout, training=self.training)
        if self.variant == "global_separated_mlp" or self.variant == "global_separated_distmult":
            assert global_state is not None
            return ego, neighbor, global_state
        if self.variant == "global_mixed_mlp":
            assert global_state is not None
            return ego + neighbor + global_state, None
        if self.variant == "mixed_mlp":
            return ego + neighbor, None
        return ego, neighbor

    def score_pairs(self, states: tuple[Tensor, ...], pairs: Tensor) -> Tensor:
        ego = states[0]
        neighbor = states[1] if len(states) > 1 else None
        left = pairs[:, 0]
        right = self.num_m + pairs[:, 1]
        if self.variant in {"sgc_mlp", "sgformer_mlp"}:
            features = self._pair_features(ego[left], ego[right])
            return self.decoder(features).squeeze(-1)
        if self.variant == "ego_only_mlp":
            features = self._pair_features(ego[left], ego[right])
            return self.decoder(features).squeeze(-1)
        if self.variant == "mixed_mlp":
            features = self._pair_features(ego[left], ego[right])
            return self.decoder(features).squeeze(-1)
        if self.variant == "global_mixed_mlp":
            features = self._pair_features(ego[left], ego[right])
            return self.decoder(features).squeeze(-1)
        if self.variant in {"global_separated_mlp", "global_separated_distmult"}:
            assert neighbor is not None
            global_state = states[2]
            if self.variant == "global_separated_distmult":
                ego_term = (ego[left] * ego[right] * self.ego_weight).sum(-1)
                neighbor_term = (neighbor[left] * neighbor[right] * self.neighbor_weight).sum(-1)
                global_term = (global_state[left] * global_state[right] * self.global_weight).sum(-1)
                return ego_term + neighbor_term + global_term + self.decoder_bias
            features = torch.cat((self._pair_features(ego[left], ego[right]),
                                  self._pair_features(neighbor[left], neighbor[right]),
                                  self._pair_features(global_state[left], global_state[right])), dim=-1)
            return self.decoder(features).squeeze(-1)
        assert neighbor is not None
        if self.variant == "separated_distmult":
            ego_term = (ego[left] * ego[right] * self.ego_weight).sum(-1)
            neighbor_term = (neighbor[left] * neighbor[right] * self.neighbor_weight).sum(-1)
            return ego_term + neighbor_term + self.decoder_bias
        features = torch.cat((self._pair_features(ego[left], ego[right]),
                              self._pair_features(neighbor[left], neighbor[right])), dim=-1)
        return self.decoder(features).squeeze(-1)

    def forward(self, metabolite_features: Tensor, disease_features: Tensor,
                edge_index: Tensor, pairs: Tensor) -> Tensor:
        states = self.encode(metabolite_features, disease_features, edge_index)
        return self.score_pairs(states, pairs)

