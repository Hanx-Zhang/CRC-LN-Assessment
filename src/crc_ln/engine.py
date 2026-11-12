"""Shared patient-level objective, deterministic sampling and inference."""
import random
import numpy as np
import torch
from torch.nn import functional as F


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


class Sampler:
    def __init__(self, rows, stage, seed):
        self.rng=random.Random(seed);self.stage=stage
        eligible=[r for r in rows if r.get('ct') or (stage==2 and r.get('optical'))]
        if stage==1:self.pools={'all':[r['patient_id'] for r in eligible if r.get('ct')]}
        else:
            self.pools={'paired':[r['patient_id'] for r in eligible if r.get('ct') and r.get('optical')],
                        'single':[r['patient_id'] for r in eligible if bool(r.get('ct')) != bool(r.get('optical'))]}
        self.pools = {k:v for k,v in self.pools.items() if v}
        if not self.pools:raise ValueError('training pool is empty')
        self.order={k:[] for k in self.pools};self.cursor={k:0 for k in self.pools}

    def take(self, pool, n):
        result=[]
        for _ in range(n):
            if self.cursor[pool]>=len(self.order[pool]):
                self.order[pool]=self.pools[pool].copy();self.rng.shuffle(self.order[pool]);self.cursor[pool]=0
            result.append(self.order[pool][self.cursor[pool]]);self.cursor[pool]+=1
        return result

    def batch(self):
        if self.stage == 1:return self.take('all',8)
        if len(self.pools) == 1:return self.take(next(iter(self.pools)),8)
        return self.take('paired',4)+self.take('single',4)

    def state_dict(self):
        return dict(pools=self.pools,order=self.order,cursor=self.cursor,rng=self.rng.getstate(),stage=self.stage)

    def load_state_dict(self,s):
        if s['pools']!=self.pools or s['stage']!=self.stage:raise ValueError('sampler identity mismatch')
        self.order=s['order'];self.cursor=s['cursor'];self.rng.setstate(s['rng'])


def patient_loss(model, sample, label, stage, weight=1.):
    out=model(sample,stage)
    if out['status']!='OK':return None
    def bce(z):return F.binary_cross_entropy_with_logits(z,z.new_tensor(float(label)))
    if stage==1:base=bce(out['ct'])
    elif out['ct'] is not None and out['wno'] is not None:
        base=(bce(out['ct'])+bce(out['joint'])+.25*bce(out['wno']))/2.25
    else:base=bce(out['ct'] if out['ct'] is not None else out['wno'])
    residuals=out['residuals']
    reg=torch.stack([r.square().mean() for r in residuals]).mean() if residuals else base.new_zeros(())
    return weight*(base+1e-5*reg)


@torch.no_grad()
def predict_rows(model, dataset, rows, stage):
    model.eval();records=[]
    for row in rows:
        sample=dataset.sample(row);outputs={}
        for name,s in [('all',sample)]:
            out=model(s,stage)
            if out['status']!='OK':outputs[name]={'status':out['status']};continue
            result={'status':'OK'}
            for public,internal in [('ct','ct'),('optical','wno'),('joint','joint')]:
                z=out[internal]
                result[public]=None if z is None else {'logit':float(z),'probability':float(z.sigmoid())}
            result['joint_route']='joint' if out['ct'] is not None and out['wno'] is not None else 'ct' if out['ct'] is not None else 'optical'
            outputs[name]=result
        records.append({'patient_id':row['patient_id'],'outputs':outputs})
    return records
