"""Diagnostic model with phase-aware graphs and modality fallback."""
import torch
from torch import nn
from ._components import UnifiedModel


class Model(UnifiedModel):
    def __init__(self, imagenet_path=None, fusion='graph'):
        if fusion not in ('graph', 'equal', 'weighted'):
            raise ValueError('unsupported model configuration')
        super().__init__(imagenet_path=imagenet_path)
        self.fusion_alpha = nn.Parameter(torch.zeros(()))
        self.model_config = dict(fusion=fusion)

    def forward(self, sample, stage=2):
        out = super().forward(sample, stage)
        if out['status'] == 'OK' and out['ct'] is not None and out['wno'] is not None:
            alpha = .5 if self.model_config['fusion'] == 'equal' else self.fusion_alpha.sigmoid()
            delta = out['residuals'][-1] if self.model_config['fusion'] == 'graph' else out['ct'].new_zeros(())
            if self.model_config['fusion'] != 'graph':
                out['residuals'] = out['residuals'][:-1]
            out['joint'] = alpha * out['ct'] + (1-alpha) * out['wno'] + delta
        return out


def parameter_groups(model, stage):
    groups = {}
    unused = ('fusion_gate.', 'fusion_residual.', 'icg.white_gate.', 'icg.white_residual.', 'icg.overlay_gate.', 'icg.overlay_residual.')
    for name, p in model.named_parameters():
        lr = None
        if stage == 1 and name.startswith(('ct.', 'ct_graph.', 'route_adapter.')):
            lr = 1e-4
        elif stage == 2:
            if name.startswith(('ct.', 'ct_graph.', 'route_adapter.')):
                if '.ln_encoder.' not in name or '.blocks.3.' in name:
                    lr = 1e-5
            elif name.startswith('icg.'):
                if '.features.' not in name:
                    lr = 1e-4
                elif '.features.7.' in name:
                    lr = 1e-5
            elif name.startswith(('cross_graph.', 'fusion_adapter.')) and model.model_config['fusion'] == 'graph':
                lr = 1e-4
            elif name == 'fusion_alpha' and model.model_config['fusion'] != 'equal':
                lr = 1e-3
        if name.startswith(unused):
            lr = None
        p.requires_grad_(lr is not None)
        if lr is not None:
            groups.setdefault(lr, []).append(p)
    return [dict(params=params, lr=lr) for lr, params in groups.items()]
