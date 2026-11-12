"""Output-specific validation threshold estimation."""
import numpy as np

def youden(y, p, role):
    if role != 'validation':
        raise ValueError('threshold fitting requires validation role')
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    if len(set(y)) != 2:
        raise ValueError('both classes required for threshold fitting')
    if len(y)!=len(p) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('invalid threshold inputs')
    u = np.unique(p)
    candidates = np.unique(np.r_[0., .5, 1., (u[1:]+u[:-1])/2])
    npos, nneg = int((y==1).sum()), int((y==0).sum())
    def key(t):
        pred = p>=t
        tp = int((pred&(y==1)).sum()); tn = int((~pred&(y==0)).sum())
        return (-(tp*nneg + tn*npos - npos*nneg), abs(t-.5), t)
    return float(min(candidates, key=key))

