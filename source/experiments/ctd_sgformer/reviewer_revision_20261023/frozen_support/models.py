"""Feature-based CTD link encoders and a faithful SGFormer adaptation.

Adapted SGFormer components originate from Qitian Wu's NeurIPS 2023 code:
https://github.com/qitianwu/SGFormer/blob/
3578e101c701491ce068bf26a9b029d2134903be/large/ours.py

MIT License
Copyright (c) 2025 Qitian Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

No learned node-ID table is used. The only embedding table identifies the two
node *types*. Context/query construction and the shared link decoder belong to
the runner, not these encoders.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


METHODS = frozenset({
    "mlp", "sgc", "gatv2", "sgformer", "full", "fixed_budget",
    "gate_uniform", "fixed_uniform", "degree_budget", "global_only",
})
LEARNED_GATE_METHODS = frozenset({"full", "gate_uniform"})
SGFORMER_METHODS = METHODS - frozenset({"mlp", "sgc", "gatv2"})


def normalized_graph_aggregate(x: Tensor, edge_index: Tensor) -> Tensor:
    """Compute the author's A_norm @ x without torch_sparse extensions.

    edge_index[0] is source and edge_index[1] is destination. Incoming degrees
    reproduce degree(col, N) in the original implementation; the runner must
    supply both directions for the undirected target-relation graph. No loops
    or edges are added here. Repeated edges accumulate just as sparse matmul.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.dtype != torch.long:
        raise ValueError("edge_index must use torch.long indices")
    n = x.shape[0]
    src, dst = edge_index
    if src.numel() == 0:
        return x * 0.0
    degree = torch.bincount(dst, minlength=n).to(dtype=torch.float32)
    inv_in = degree[dst].reciprocal().sqrt()
    inv_out = degree[src].reciprocal().sqrt()
    weight = torch.nan_to_num(inv_in * inv_out, nan=0.0, posinf=0.0, neginf=0.0)
    messages = x[src] * weight.to(dtype=x.dtype).unsqueeze(-1)
    return torch.zeros_like(x).index_add(0, dst, messages)


class GraphConvLayer(nn.Module):
    """The weighted, non-initial-concatenation author graph layer."""

    def __init__(self, channels: int):
        super().__init__()
        self.W = nn.Linear(channels, channels)

    def reset_parameters(self) -> None:
        self.W.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.W(normalized_graph_aggregate(x, edge_index))


class GraphConv(nn.Module):
    """Author large/ours.py graph branch with defaults enabled.

    The original branch does NOT append later states to its layer_ list. Thus
    every graph layer adds the initial projected state, rather than changing
    this into a different skip architecture when num_layers > 1.
    """

    def __init__(self, in_channels: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(in_channels, hidden)])
        self.convs = nn.ModuleList([GraphConvLayer(hidden) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers + 1)])
        self.dropout = dropout

    def reset_parameters(self) -> None:
        for module in [*self.fcs, *self.convs, *self.bns]:
            module.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.fcs[0](x)
        x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        initial_state = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + initial_state
        return x


class TransConvLayer(nn.Module):
    """One-head author global attention with safe whole-tensor norms."""

    def __init__(self, channels: int):
        super().__init__()
        self.Wk = nn.Linear(channels, channels)
        self.Wq = nn.Linear(channels, channels)
        self.Wv = nn.Linear(channels, channels)

    def reset_parameters(self) -> None:
        for layer in (self.Wk, self.Wq, self.Wv):
            layer.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        # Float32 calculations keep the author's global Frobenius normalization
        # (not PyG's per-node normalization), while avoiding a zero norm.
        qs = self.Wq(x).float().unsqueeze(1)
        ks = self.Wk(x).float().unsqueeze(1)
        vs = self.Wv(x).float().unsqueeze(1)
        qs = qs / torch.norm(qs, p=2).clamp_min(1e-8)
        ks = ks / torch.norm(ks, p=2).clamp_min(1e-8)
        n = qs.shape[0]
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        numerator = torch.einsum("nhm,hmd->nhd", qs, kvs) + n * vs
        ks_sum = ks.sum(dim=0)
        denominator = torch.einsum("nhm,hm->nh", qs, ks_sum).unsqueeze(-1) + n
        result = numerator / denominator
        return result.mean(dim=1).to(dtype=x.dtype)


class TransConv(nn.Module):
    """Author projected global branch, fixed to one attention layer."""

    def __init__(self, in_channels: int, hidden: int, dropout: float):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(in_channels, hidden)])
        self.convs = nn.ModuleList([TransConvLayer(hidden)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden)])
        self.dropout = dropout

    def reset_parameters(self) -> None:
        for module in [*self.fcs, *self.convs, *self.bns]:
            module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bns[0](self.fcs[0](x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        previous = x
        x = self.convs[0](x)
        x = (x + previous) / 2.0
        x = F.relu(self.bns[1](x))
        return F.dropout(x, p=self.dropout, training=self.training)


class SGFormerCore(nn.Module):
    """SGFormer raw 64-D projection; no classification softmax.

    use_graph=False genuinely omits graph_conv, unlike the original flag.
    Node-dependent mixing weights implement the explicit research change.
    """

    def __init__(self, in_channels: int = 128, hidden: int = 64,
                 layers: int = 1, dropout: float = 0.2,
                 alpha: float = 0.5, use_graph: bool = True):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden, dropout)
        self.use_graph = use_graph
        if use_graph:
            self.graph_conv = GraphConv(in_channels, hidden, layers, dropout)
        self.fc = nn.Linear(hidden, 64)
        self.alpha = alpha

    def reset_parameters(self) -> None:
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
        # The author's reset omitted this head; reset the entire adaptation.
        self.fc.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor,
                mixing_weight: Optional[Tensor] = None) -> Tensor:
        global_features = self.trans_conv(x)
        if self.use_graph:
            local_features = self.graph_conv(x, edge_index)
            weight = self.alpha if mixing_weight is None else mixing_weight
            features = weight * local_features + (1.0 - weight) * global_features
        else:
            features = global_features
        return self.fc(features)


class TypeFeatureAdapter(nn.Module):
    """Identical raw attribute access and frontend for every method."""

    def __init__(self, metabolite_dim: int, disease_dim: int,
                 num_m: int, num_d: int):
        super().__init__()
        self.metabolite_projection = nn.Linear(metabolite_dim, 128)
        self.disease_projection = nn.Linear(disease_dim, 128)
        self.type_embedding = nn.Embedding(2, 128)
        self.num_m = num_m
        self.num_d = num_d
        self.metabolite_dim = metabolite_dim
        self.disease_dim = disease_dim
        self.register_buffer("node_types", torch.cat([
            torch.zeros(num_m, dtype=torch.long),
            torch.ones(num_d, dtype=torch.long),
        ]))

    def forward(self, metabolite_features: Tensor, disease_features: Tensor) -> Tensor:
        if metabolite_features.shape != (self.num_m, self.metabolite_dim):
            raise ValueError("metabolite_features do not match the declared shape")
        if disease_features.shape != (self.num_d, self.disease_dim):
            raise ValueError("disease_features do not match the declared shape")
        attributes = torch.cat([
            self.metabolite_projection(metabolite_features),
            self.disease_projection(disease_features),
        ], dim=0)
        # Add types within 128 dimensions; do not concatenate or use node IDs.
        return attributes + self.type_embedding(self.node_types)

    def reset_parameters(self) -> None:
        self.metabolite_projection.reset_parameters()
        self.disease_projection.reset_parameters()
        self.type_embedding.reset_parameters()


class FeatureLinkEncoder(nn.Module):
    """Uniform model interface; edge masking is controlled by the runner."""

    def __init__(self, method: str, metabolite_dim: int, disease_dim: int,
                 num_m: int, num_d: int, hidden: int, layers: int,
                 dropout: float, alpha: float, c: float):
        super().__init__()
        self.method = method
        self.adapter = TypeFeatureAdapter(metabolite_dim, disease_dim, num_m, num_d)
        self.num_nodes = num_m + num_d
        self.c = c
        self.alpha = alpha
        self.last_gate: Optional[Tensor] = None

        if method == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(128, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 64),
            )
        elif method == "sgc":
            from torch_geometric.nn import SGConv
            self.encoder = SGConv(128, 64, K=layers, cached=False,
                                  add_self_loops=True, bias=True)
        elif method == "gatv2":
            from torch_geometric.nn import GATv2Conv
            dims = [128] + [hidden] * (layers - 1) + [64]
            self.convs = nn.ModuleList([
                GATv2Conv(dims[i], dims[i + 1], heads=1, concat=False,
                          dropout=dropout, add_self_loops=True, bias=True)
                for i in range(layers)
            ])
            self.dropout = dropout
        else:
            self.encoder = SGFormerCore(
                in_channels=128, hidden=hidden, layers=layers,
                dropout=dropout, alpha=alpha, use_graph=method != "global_only",
            )
            if method in LEARNED_GATE_METHODS:
                self.gate = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))

    def _mixing_weight(self, degree: Tensor) -> Tensor:
        degree = degree.reshape(-1, 1).float()
        if self.method in LEARNED_GATE_METHODS:
            type_one_hot = F.one_hot(self.adapter.node_types, num_classes=2).float()
            gate_input = torch.cat([torch.log1p(degree), type_one_hot], dim=1)
            weight = (degree > 0).float() * torch.sigmoid(self.gate(gate_input))
        elif self.method == "degree_budget":
            weight = degree / (degree + self.c)
        else:
            weight = torch.full_like(degree, self.alpha)
        self.last_gate = weight.detach()
        return weight

    def forward(self, metabolite_features: Tensor, disease_features: Tensor,
                edge_index: Tensor, degree: Tensor) -> Tensor:
        if degree.numel() != self.num_nodes:
            raise ValueError("degree must contain one current target-relation degree per node")
        if not torch.isfinite(degree).all() or (degree < 0).any():
            raise ValueError("degree must contain finite nonnegative counts")
        x = self.adapter(metabolite_features, disease_features)
        if self.method == "mlp":
            return self.encoder(x)
        if self.method == "sgc":
            return self.encoder(x, edge_index)
        if self.method == "gatv2":
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
            return x
        if self.method == "global_only":
            self.last_gate = None
            return self.encoder(x, edge_index)
        return self.encoder(x, edge_index, self._mixing_weight(degree))

    def reset_parameters(self) -> None:
        self.adapter.reset_parameters()
        if self.method == "mlp":
            for layer in self.encoder:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        elif self.method == "gatv2":
            for layer in self.convs:
                layer.reset_parameters()
        else:
            self.encoder.reset_parameters()
        if self.method in LEARNED_GATE_METHODS:
            for layer in self.gate:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        self.last_gate = None


def create_model(method: str, metabolite_dim: int, disease_dim: int,
                 num_m: int, num_d: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.2, alpha: float = 0.5,
                 c: float = 1.0) -> nn.Module:
    """Create a raw-embedding encoder. Decoder and masks are runner-owned.

    layers controls local SGFormer depth, SGC hops or GATv2 depth; SGFormer
    global attention stays at one head and one layer. MLP uses two linears.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")
    if any(v <= 0 for v in (metabolite_dim, disease_dim, num_m, num_d, hidden, layers)):
        raise ValueError("feature dimensions, node counts, hidden and layers must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1 for both active branches")
    if c <= 0:
        raise ValueError("degree-rule c must be positive")
    return FeatureLinkEncoder(method, metabolite_dim, disease_dim, num_m, num_d,
                              hidden, layers, dropout, alpha, c)
