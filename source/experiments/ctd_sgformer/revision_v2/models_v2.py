"""Review-driven SGFormer controls; original frozen implementation is unchanged."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from models import create_model, FeatureLinkEncoder

class RevisedEncoder(FeatureLinkEncoder):
    def __init__(self, variant, *args, **kwargs):
        # Same modules and initialization as the original learned-gate encoder.
        super().__init__('full', *args, **kwargs)
        self.variant = variant
        if variant == 'residual_fixed':
            del self.gate

    def _mixing_weight(self, degree):
        degree = degree.reshape(-1, 1).float()
        types = F.one_hot(self.adapter.node_types, 2).float()
        if self.variant == 'residual_fixed':
            gate = torch.ones_like(degree)
        else:
            gate = torch.sigmoid(self.gate(torch.cat([torch.log1p(degree), types], 1)))
        if self.variant != 'soft':
            gate = gate * (degree > 0).float()
        self.last_gate = gate.detach()
        return gate

    def forward(self, fm, fd, edge_index, degree):
        if self.variant == 'soft':
            return super().forward(fm, fd, edge_index, degree)
        if degree.numel() != self.num_nodes or not torch.isfinite(degree).all() or (degree < 0).any():
            raise ValueError('degree must contain one finite nonnegative count per node')
        if len(self.encoder.graph_conv.convs) != 1:
            raise ValueError('Residual-preserving design is defined for one local layer.')
        x = self.adapter(fm, fd)
        global_h = self.encoder.trans_conv(x)
        local = self.encoder.graph_conv
        residual = F.dropout(F.relu(local.bns[0](local.fcs[0](x))),
                             p=local.dropout, training=self.training)
        message = F.dropout(F.relu(local.bns[1](local.convs[0](residual, edge_index))),
                            p=local.dropout, training=self.training)
        gate = self._mixing_weight(degree)
        # Keep attribute processing at k=0; suppress only neighbor-message branch.
        mixed = (1 - self.alpha) * global_h + self.alpha * residual + self.alpha * gate * message
        return self.encoder.fc(mixed)

    def reset_parameters(self):
        self.adapter.reset_parameters()
        self.encoder.reset_parameters()
        if hasattr(self, 'gate'):
            for layer in self.gate:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self.last_gate = None

def make_model(method, cfg, fm, fd):
    args = (fm.shape[1], fd.shape[1], fm.shape[0], fd.shape[0])
    kw = dict(hidden=64, layers=1, dropout=0.2, alpha=cfg['alpha'], c=1.0)
    if method in ('soft_budget', 'residual_budget', 'residual_none', 'residual_fixed_budget'):
        variant = 'soft' if method == 'soft_budget' else 'residual_fixed' if method == 'residual_fixed_budget' else 'residual'
        return RevisedEncoder(variant, *args, **kw)
    mapping = {'fixed_none':'sgformer', 'fixed_budget':'sgformer', 'fixed_uniform':'sgformer',
               'fixed_tuned':'sgformer', 'hard_none':'full', 'hard_budget':'full', 'hard_uniform':'full'}
    return create_model(mapping.get(method, method), *args, **kw)
