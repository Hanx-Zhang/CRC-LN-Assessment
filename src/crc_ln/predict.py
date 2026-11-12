"""Label-free prediction with checkpoint-bound thresholds."""
import argparse
import json
from pathlib import Path
import torch
from .model import Model
from .data import Dataset,sha256,write_json
from .engine import predict_rows


def load_model(checkpoint,device='cpu'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Model(**state['model_config']).to(device)
    model.load_state_dict(state['model'],strict=True);model.eval()
    return model,state


def run(checkpoint,manifest,threshold_file,output,role='test',device='cpu'):
    torch.set_num_threads(2)
    thresholds=json.loads(Path(threshold_file).read_text(encoding='utf-8'))
    if thresholds['checkpoint_sha256']!=sha256(checkpoint) or thresholds['role']!='validation':
        raise ValueError('threshold/model identity mismatch')
    model,state=load_model(checkpoint,device);data=Dataset(manifest)
    records=predict_rows(model,data,data.role(role),state['stage'])
    # Branch fallback uses that branch's own frozen threshold.
    for r in records:
        for out in r['outputs'].values():
            if out['status']!='OK':continue
            for branch in ('ct','optical','joint'):
                if out[branch] is None:continue
                route=out['joint_route'] if branch=='joint' else branch
                t=thresholds['thresholds'][route]
                out[branch]['threshold']=t
                out[branch]['class']=None if t is None else int(out[branch]['probability']>=t)
    write_json(output,dict(checkpoint_sha256=sha256(checkpoint),manifest_sha256=data.sha,role=role,
                          threshold_sha256=sha256(threshold_file),records=records))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--thresholds',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--role',choices=['train','validation','test'],default='test');a=p.parse_args()
    run(a.checkpoint,a.manifest,a.thresholds,a.output,a.role,a.device)

if __name__=='__main__':main()
