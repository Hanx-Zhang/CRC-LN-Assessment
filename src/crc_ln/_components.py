"""Numeric components for phase-aware CT and optical multiple-instance learning."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18, resnet34
PHASES = ['A', 'V', 'NCCT', 'DELAYED', 'ENHANCED_UNKNOWN', 'UNKNOWN']

class ResidualBlock(nn.Module):

    def __init__(self, inp: int, out: int, stride: int=1) -> None:
        super().__init__()
        self.c1 = nn.Conv3d(inp, out, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, out)
        self.c2 = nn.Conv3d(out, out, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, out)
        self.skip = nn.Identity() if inp == out and stride == 1 else nn.Sequential(nn.Conv3d(inp, out, 1, stride, bias=False), nn.GroupNorm(8, out))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.n2(self.c2(self.act(self.n1(self.c1(x))))) + self.skip(x))

class LNEncoder(nn.Module):

    def __init__(self, channels: int=2, base: int=32) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(channels, base, 3, 1, 1, bias=False), nn.GroupNorm(8, base), nn.GELU())
        self.blocks = nn.Sequential(ResidualBlock(base, base), ResidualBlock(base, base * 2, 2), ResidualBlock(base * 2, base * 4, 2), ResidualBlock(base * 4, base * 8, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.embedding_dim = base * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(x))).flatten(1)


class GatedAttention(nn.Module):

    def __init__(self, dim: int=256, hidden: int=128) -> None:
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        raw = raw.masked_fill(~mask, -torch.inf)
        weight = torch.softmax(raw, dim=1)
        weight = torch.where(mask, weight, torch.zeros_like(weight))
        return (torch.sum(weight.unsqueeze(-1) * x, dim=1), weight)

class PatientMIL(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln_encoder = LNEncoder(channels=2, base=32)
        self.station_embedding = nn.Embedding(4, 16, padding_idx=0)
        self.station_projection = nn.Linear(272, 256)
        self.empty_token = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.normal_(self.empty_token, std=0.02)
        self.attention = GatedAttention(256, 128)
        self.patient_head = nn.Linear(256, 1)

    def _pool(self, x, mask):
        return self.attention(x, mask)

class PhaseMaskedAVMIL(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = PatientMIL()
        self.v = PatientMIL()
        self.phase_gate = nn.Sequential(nn.Linear(514, 128), nn.GELU(), nn.Linear(128, 2))
        self.interaction = nn.Linear(512, 1)
        nn.init.zeros_(self.interaction.weight)
        nn.init.zeros_(self.interaction.bias)


class ImageEncoder(nn.Module):

    def __init__(self, backbone: str, channels: int, weights: Path | None, dim: int=256):
        super().__init__()
        constructors = {'r18': resnet18, 'r34': resnet34}
        network = constructors[backbone](weights=None)
        if weights is not None:
            network.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        if channels == 1:
            original = network.conv1.weight.detach()
            network.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                network.conv1.weight.copy_(original.mean(1, keepdim=True))
        self.features = nn.Sequential(*list(network.children())[:-2])
        self.projection = nn.Linear(512, dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        outputs = []
        for chunk in image.split(4):
            if self.training and any((p.requires_grad for p in self.features.parameters())):
                feat = checkpoint(self.features, chunk, use_reentrant=False)
            else:
                feat = self.features(chunk)
            outputs.append(self.projection(F.adaptive_avg_pool2d(feat, 1).flatten(1)))
        return torch.cat(outputs)

def normalized_smoothmax(values: torch.Tensor, mask: torch.Tensor, temperature: float=0.5):
    count = mask.sum(-1).clamp_min(1)
    scaled = (values / temperature).masked_fill(~mask, -torch.inf)
    pooled = temperature * (torch.logsumexp(scaled, -1) - count.log())
    weight = torch.softmax(scaled, -1).masked_fill(~mask, 0.0)
    valid = mask.any(-1)
    return (torch.where(valid, pooled, torch.zeros_like(pooled)), torch.where(valid.unsqueeze(-1), weight, torch.zeros_like(weight)))

class ICGSmoothMaxMILV3(nn.Module):

    def __init__(self, backbone: str, modality: str, weights: Path | None, temperature: float=0.5):
        super().__init__()
        if modality not in {'nir', 'white', 'wn', 'overlay', 'wno'}:
            raise ValueError(modality)
        self.backbone = backbone
        self.modality = modality
        self.temperature = temperature
        requested = ('nir',) if modality == 'nir' else ('white',) if modality == 'white' else ('overlay',) if modality == 'overlay' else ('nir', 'white') if modality == 'wn' else ('nir', 'white', 'overlay')
        self.encoders = nn.ModuleDict({name: ImageEncoder(backbone, 1 if name == 'nir' else 3, weights) for name in requested})
        if modality in {'wn', 'wno'}:
            self.white_gate = nn.Linear(512, 1)
            self.white_residual = nn.Linear(256, 256)
            nn.init.zeros_(self.white_residual.weight)
            nn.init.zeros_(self.white_residual.bias)
        if modality == 'wno':
            self.overlay_gate = nn.Linear(512, 1)
            self.overlay_residual = nn.Linear(256, 256)
            nn.init.zeros_(self.overlay_residual.weight)
            nn.init.zeros_(self.overlay_residual.bias)
        self.instance_head = nn.Linear(256, 1)

    def train(self, mode: bool=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


    @staticmethod
    def _encode(encoder: ImageEncoder, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        flat = image.reshape(-1, *image.shape[-3:])
        valid = mask.reshape(-1)
        output = flat.new_zeros((len(flat), 256))
        if valid.any():
            idx = valid.nonzero(as_tuple=False).squeeze(1)
            output = output.index_copy(0, idx, encoder(flat[idx]).to(output.dtype))
        return output.reshape(*mask.shape, 256)


class MessageGraph(nn.Module):

    def __init__(self, dim=256, hidden=64):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.phase = nn.Embedding(7, hidden)
        self.msg = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Sequential(nn.Linear(hidden * 2, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, nodes, phases):
        if len(nodes) == 0:
            return (None, 0)
        h = torch.tanh(self.proj(nodes) + self.phase(phases))
        if len(h) == 1:
            return (h.mean(0), 0)
        sim = F.normalize(h, dim=-1) @ F.normalize(h, dim=-1).T
        sim = sim.masked_fill(torch.eye(len(h), device=h.device, dtype=torch.bool), -torch.inf)
        k = min(2, len(h) - 1)
        ix = sim.topk(k, dim=1).indices
        pair = torch.cat([h[:, None, :].expand(-1, k, -1), h[ix]], -1)
        w = torch.softmax(self.edge(pair).squeeze(-1), -1)
        return ((h + (w[..., None] * self.msg(h[ix])).sum(1)).mean(0), len(h) * k)

class CrossGraph(nn.Module):

    def __init__(self):
        super().__init__()
        self.ct = nn.Linear(256, 64)
        self.wno = nn.Linear(256, 64)
        self.ct_to_wno = nn.Linear(64, 64, bias=False)
        self.wno_to_ct = nn.Linear(64, 64, bias=False)
        self.score_c = nn.Linear(128, 1)
        self.score_w = nn.Linear(128, 1)

    def forward(self, c, w):
        if not len(c) or not len(w):
            return (None, 0)
        c = torch.tanh(self.ct(c))
        w = torch.tanh(self.wno(w))
        sim = F.normalize(c, dim=-1) @ F.normalize(w, dim=-1).T
        cw = sim.topk(min(2, len(w)), dim=1).indices
        wc = sim.T.topk(min(2, len(c)), dim=1).indices
        a = torch.softmax(self.score_c(torch.cat([c[:, None, :].expand(-1, cw.shape[1], -1), w[cw]], -1)).squeeze(-1), -1)
        b = torch.softmax(self.score_w(torch.cat([w[:, None, :].expand(-1, wc.shape[1], -1), c[wc]], -1)).squeeze(-1), -1)
        cc = c + (a[..., None] * self.wno_to_ct(w[cw])).sum(1)
        ww = w + (b[..., None] * self.ct_to_wno(c[wc])).sum(1)
        return ((cc.mean(0) + ww.mean(0)) / 2, cw.numel() + wc.numel())

class RouteAdapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.down = nn.Linear(72, 8)
        self.up = nn.Linear(8, 1)
        self.gate = nn.Linear(72, 1)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, summary, route):
        x = torch.cat([summary, route])
        return 0.1 * torch.sigmoid(self.gate(x)) * torch.tanh(self.up(self.down(x)))

class UnifiedModel(nn.Module):

    def __init__(self, imagenet_path=None, ct_graph_enabled=True, cross_graph_enabled=True):
        super().__init__()
        self.ct = PhaseMaskedAVMIL()
        self.ct_graph_enabled = ct_graph_enabled
        self.cross_graph_enabled = cross_graph_enabled
        self.icg = Optical('r34', 'wno', imagenet_path)
        self.ct_graph = MessageGraph()
        self.cross_graph = CrossGraph()
        self.route_adapter = RouteAdapter()
        self.fusion_adapter = RouteAdapter()
        self.fusion_gate = nn.Linear(513, 1)
        self.fusion_residual = nn.Linear(256, 1)
        nn.init.zeros_(self.fusion_residual.weight)
        nn.init.zeros_(self.fusion_residual.bias)
    def branch(self, phase, patches, station):
        mil = getattr(self.ct, phase)
        if not len(patches):
            raise RuntimeError('empty_bag_must_not_enter_MIL')
        emb = []
        for x in patches.split(4):
            if self.training and any((p.requires_grad for p in mil.ln_encoder.parameters())):
                emb.append(checkpoint(mil.ln_encoder, x, use_reentrant=False))
            else:
                emb.append(mil.ln_encoder(x))
        e = mil.station_projection(torch.cat([torch.cat(emb), mil.station_embedding(station.clamp(0, 3))], -1))
        bag, att = mil._pool(e[None], torch.ones((1, len(e)), device=e.device, dtype=torch.bool))
        return (mil.patient_head(bag).reshape(()), bag[0], e, att[0])

    def ct_case_group(self, cases, device):
        outputs = {}
        all_nodes = []
        pids = []
        edges = 0
        for ph in ('a', 'v'):
            selected = [c for c in cases if ('a' if c['phase_id'] == 'A' else 'v') == ph]
            if not selected:
                continue
            records = []
            for c in selected:
                for j, uid in enumerate(c['uids']):
                    if not str(uid).strip() or ('node_mask' in c and (not bool(c['node_mask'][j]))):
                        continue
                    records.append((uid, c, j))
            if not records:
                continue
            records.sort(key=lambda x: x[0])
            patches = torch.stack([c['patches'][j] for _, c, j in records]).to(device)
            station = torch.stack([c['station'][j] for _, c, j in records]).to(device)
            z, bag, nodes, att = self.branch(ph, patches, station)
            phases = torch.tensor([PHASES.index(c['phase_id']) for _, c, j in records], device=device)
            all_nodes.append(nodes[:4])
            pids.append(phases[:4])
            outputs[ph] = (z, bag)
        if not outputs:
            return (None, None, None, None, 0)
        ref = next(iter(outputs.values()))[0]
        zero = ref.new_zeros(256)
        aa = 'a' in outputs
        vv = 'v' in outputs
        ae = outputs['a'][1] if aa else zero
        ve = outputs['v'][1] if vv else zero
        av = ref.new_tensor([aa, vv])
        features = torch.cat([ae, ve, av])
        weight = torch.softmax(self.ct.phase_gate(features).masked_fill(~av.bool(), -torch.inf), 0)
        z = sum((weight[i] * outputs[p][0] for i, p in enumerate(('a', 'v')) if p in outputs))
        if aa and vv:
            z = z + torch.tanh(self.ct.interaction(torch.cat([ae, ve]))).reshape(())
        pooled = weight[0] * ae + weight[1] * ve
        nodes = torch.cat(all_nodes)
        phases = torch.cat(pids)
        summary, edges = self.ct_graph(nodes, phases)
        route = torch.cat([F.one_hot(phases, num_classes=6).float().mean(0), av])
        delta = self.route_adapter(summary, route).reshape(()) if self.ct_graph_enabled else z.new_zeros(())
        return (z + delta, pooled, nodes, delta, edges)

    def forward(self, sample, stage=2):
        device = next(self.parameters()).device
        cases = sample['cases']
        series = [c.get('series_uid') or c.get('case_id') for c in cases]
        if len(series) != len(set(series)):
            raise ValueError('duplicate_series_uid')
        groups = [[c] for c in cases] if sample.get('ct_aggregation') == 'sequence_mean' else [cases] if cases else []
        zs = []
        embs = []
        nodes = []
        ds = []
        edge_count = 0
        for group in groups:
            z, e, n, d, edges = self.ct_case_group(group, device)
            if z is not None:
                zs.append(z)
                embs.append(e)
                nodes.append(n)
                ds.append(d)
                edge_count += edges
        ct = torch.stack(zs).mean() if zs else None
        ct_e = torch.stack(embs).mean(0) if embs else None
        wno = None
        cross_delta = None
        wno_nodes = 0
        if stage == 2 and sample.get('icg') is not None:
            b = sample['icg']
            images = {k: v.unsqueeze(0).to(device) for k, v in b['images'].items()}
            valid = {k: v.unsqueeze(0).to(device) for k, v in b['valid'].items()}
            io = self.icg(images, valid, b['frame_mask'][None].to(device), b['candidate_mask'][None].to(device))
            if io['pooling_mask'].any():
                wno = io['logit'].reshape(())
                wn = []
                for f in range(io['pooling_mask'].shape[1]):
                    ix = io['pooling_mask'][0, f].nonzero().flatten()
                    if len(ix):
                        weights = io['candidate_weight'][0, f, ix]
                        wn.append((io['instance_embedding'][0, f, ix] * weights[:, None]).sum(0))
                wn = torch.stack(wn)
                wno_nodes = len(wn)
                if ct is not None:
                    cn = torch.cat(nodes)
                    summary, ne = self.cross_graph(cn, wn)
                    edge_count += ne
                    phase_count = torch.zeros(6, device=device)
                    for c in cases:
                        phase_count[PHASES.index(c['phase_id'])] += 1
                    phase_count /= phase_count.sum().clamp_min(1)
                    route = torch.cat([phase_count, phase_count.new_tensor([1.0, 1.0])])
                    cross_delta = self.fusion_adapter(summary, route).reshape(()) if self.cross_graph_enabled else ct.new_zeros(())
                    gate = torch.sigmoid(self.fusion_gate(torch.cat([ct_e, io['patient_embedding'][0], ct_e.new_ones(1)]))).reshape(())
                    joint = ct + gate * torch.tanh(self.fusion_residual(io['patient_embedding'][0])).reshape(()) + cross_delta
                else:
                    joint = wno
        if wno is None:
            joint = ct
        if joint is None:
            return dict(status='REVIEW_NO_PREDICTION')
        residuals = ds + ([cross_delta] if cross_delta is not None else [])
        return dict(status='OK', joint=joint, ct=ct, wno=wno, residuals=residuals, nodes=sum((len(n) for n in nodes)) + wno_nodes, edges=edge_count, missing_wno=wno is None, ct_case_count=len(zs))

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

class Optical(ICGSmoothMaxMILV3):
    def forward(self, images, valid, frame_mask, candidate_mask):
        masks = {k: candidate_mask & valid[k] for k in self.encoders}
        count = torch.stack(list(masks.values())).sum(0)
        fused = sum(self._encode(self.encoders[k], images[k], masks[k]) for k in masks)
        fused = fused / count.clamp_min(1).unsqueeze(-1)
        mask = count > 0
        il = self.instance_head(fused).squeeze(-1)
        fl, cw = normalized_smoothmax(il, mask, self.temperature)
        pl, fw = normalized_smoothmax(fl, frame_mask & mask.any(-1), self.temperature)
        return dict(logit=pl, patient_embedding=((fused * cw.unsqueeze(-1)).sum(-2) * fw.unsqueeze(-1)).sum(-2),
                    instance_logit=il, instance_embedding=fused, candidate_weight=cw, frame_weight=fw,
                    hierarchical_weight=cw * fw.unsqueeze(-1), pooling_mask=mask)
