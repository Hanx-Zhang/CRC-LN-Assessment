"""Patient-bound candidate inputs; labels are never included in model inputs."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ._components import PHASES


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


class Dataset:
    def __init__(self, manifest, verify=True):
        self.path = Path(manifest).resolve()
        self.sha = sha256(self.path)
        payload = json.loads(self.path.read_text(encoding='utf-8'))
        self.rows = payload['patients']
        self.by_id = {}
        for row in self.rows:
            pid = row['patient_id']
            if not pid or pid in self.by_id:
                raise ValueError('patient_id must be globally unique across roles')
            if row['split'] not in ('development', 'train', 'validation', 'test'):
                raise ValueError('explicit patient split required')
            if 'label' in row and row['label'] not in (0, 1):
                raise ValueError('binary patient label required')
            self.by_id[pid] = row
            series = set()
            for case in row.get('ct', []):
                if case['series_uid'] in series or case['phase'] not in PHASES:
                    raise ValueError('duplicate CT series or unsupported phase')
                series.add(case['series_uid'])
                if verify:
                    self.verified_file(case)
            unit = set()
            for g in row.get('optical', []):
                if g['group_uid'] in unit:
                    raise ValueError('duplicate optical group')
                unit.add(g['group_uid'])
                ids = [c['uid'] for c in g['candidates']]
                if len(ids) != len(set(ids)):
                    raise ValueError('duplicate optical candidate')
                for c in g['candidates']:
                    for name, view in c['views'].items():
                        if name not in ('white', 'nir', 'overlay'):
                            raise ValueError('unsupported optical view')
                        if verify:
                            self.verified_file(view)

    def verified_file(self, record):
        path = self.path.parent / record['path']
        if sha256(path) != record['sha256']:
            raise ValueError('input SHA256 mismatch')
        return path

    def role(self, role, require_labels=False):
        rows = [r for r in self.rows if r['split'] == role]
        if require_labels and any('label' not in r for r in rows):
            raise ValueError('labels missing for requested role')
        return rows

    def sample(self, row):
        cases = []
        for record in row.get('ct', []):
            with np.load(self.path.parent / record['path'], allow_pickle=False) as f:
                ct, mask = f['ct'].copy(), f['target_mask'].copy()
                uid = f['uids'].astype(str).tolist()
                station = f['station'].astype(np.int64)
            if ct.shape != mask.shape or ct.shape[1:] != (32, 32, 32):
                raise ValueError('CT input must be N x 32 x 32 x 32')
            if not np.isfinite(ct).all() or not np.isin(mask, [0, 1]).all():
                raise ValueError('nonfinite CT or nonbinary mask')
            if len(uid) != len(ct) or len(station) != len(ct) or len(set(uid)) != len(uid) or any(not x for x in uid):
                raise ValueError('candidate identity mismatch')
            if not np.isin(station, [0, 1, 2, 3]).all():
                raise ValueError('station outside schema')
            cases.append(dict(phase_id=record['phase'], series_uid=record['series_uid'],
                              uids=uid, patches=torch.from_numpy(np.stack([ct, mask], 1)).float(),
                              station=torch.from_numpy(station)))
        groups = sorted(row.get('optical', []), key=lambda g: g['group_uid'])
        groups = [g for g in groups if any(c['views'] for c in g['candidates'])]
        bag = None
        if groups:
            n = max(len(g['candidates']) for g in groups)
            images = {k: torch.zeros(len(groups), n, ch, 224, 224) for k, ch in [('white', 3), ('nir', 1), ('overlay', 3)]}
            valid = {k: torch.zeros(len(groups), n, dtype=torch.bool) for k in images}
            ids = []
            for i, g in enumerate(groups):
                candidates = sorted(g['candidates'], key=lambda c: c['uid'])
                ids.append([c['uid'] for c in candidates])
                for j, c in enumerate(candidates):
                    for k, view in c['views'].items():
                        with Image.open(self.path.parent / view['path']) as im:
                            arr = np.asarray(im.convert('L' if k == 'nir' else 'RGB').resize((224,224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
                        x = torch.from_numpy(arr.copy())[None] if k == 'nir' else torch.from_numpy(arr.copy()).permute(2,0,1)
                        images[k][i,j] = (x-.5)/.25 if k == 'nir' else (x-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None]
                        valid[k][i,j] = True
            cm = torch.stack(list(valid.values())).any(0)
            bag = dict(images=images, valid=valid, frame_mask=cm.any(-1), candidate_mask=cm,
                       group_uids=[g['group_uid'] for g in groups], instance_uids=ids)
        return dict(cases=cases, icg=bag, ct_aggregation=row.get('ct_aggregation', 'merged'))


def development_split(rows, fraction, seed):
    """Reserve validation patients within development; preserve test assignments."""
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    if any(r['split'] not in ('development', 'test') for r in rows):
        raise ValueError('training input roles must be development or test')
    pools = {y: [r for r in rows if r['split'] == 'development' and r.get('label') == y] for y in (0, 1)}
    development = [r for r in rows if r['split'] == 'development']
    if len(development) != sum(map(len, pools.values())) or any(len(v) < 2 for v in pools.values()):
        raise ValueError('development requires at least two patients of each class')
    target = int(len(development) * fraction + .5)
    if not 2 <= target <= len(development) - 2:
        raise ValueError('validation fraction cannot retain both classes in each split')
    n0 = max(1, min(len(pools[0])-1, int(len(pools[0])*fraction+.5)))
    n0 = max(n0, target-len(pools[1])+1)
    n0 = min(n0, target-1)
    counts = {0: n0, 1: target-n0}
    validation_ids = set()
    for label, pool in pools.items():
        ordered = sorted(pool, key=lambda r: (hashlib.sha256(f"{seed}:{r['patient_id']}".encode()).hexdigest(), r['patient_id']))
        validation_ids.update(r['patient_id'] for r in ordered[:counts[label]])
    return [dict(r, split=('test' if r['split']=='test' else 'validation' if r['patient_id'] in validation_ids else 'train')) for r in rows]


def augment(sample, rng):
    sample = copy.deepcopy(sample)
    axes = [i + 2 for i in range(3) if rng.random() < .5]
    if axes:
        for c in sample['cases']:
            c['patches'] = c['patches'].flip(axes)
    if sample['icg'] and rng.random() < .5:
        sample['icg']['images'] = {k: x.flip(-1) for k, x in sample['icg']['images'].items()}
    return sample


def mask_arterial(sample, rng, probability, stage):
    """Stage 1 missing-phase augmentation, only when both branches are valid."""
    if not 0 <= probability <= 1:
        raise ValueError('arterial dropout probability must lie in [0, 1]')
    if stage != 1 or probability == 0:
        return sample
    def valid(case):
        return any(str(uid).strip() and ('node_mask' not in case or bool(case['node_mask'][i]))
                   for i, uid in enumerate(case['uids']))
    arterial = any(c['phase_id'] == 'A' and valid(c) for c in sample['cases'])
    nonarterial = any(c['phase_id'] != 'A' and valid(c) for c in sample['cases'])
    if arterial and nonarterial and rng.random() < probability:
        return dict(sample, cases=[c for c in sample['cases'] if c['phase_id'] != 'A'])
    return sample
