"""Two-stage training with patient-separated validation and complete recovery."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from .data import Dataset, augment, mask_arterial, development_split, sha256, write_json
from .model import Model, parameter_groups
from .engine import Sampler, seed_all, rng_state, restore_rng, patient_loss, predict_rows
from .metrics import youden


def atomic_save(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(value,tmp);tmp.replace(path)


def input_weight_identity(initial, imagenet, recovery=None):
    previous = recovery['identity'] if recovery else {}
    return {name: sha256(path) if path else previous.get(name)
            for name, path in [('initial_sha256', initial), ('imagenet_sha256', imagenet)]}


def validation(model, dataset, rows, stage):
    pred=predict_rows(model,dataset,rows,stage);values=[]
    for r in pred:
        out=r['outputs']['all']
        if out['status']!='OK':continue
        z=out['joint']['logit'];y=dataset.by_id[r['patient_id']]['label']
        values.append(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(z),torch.tensor(float(y))).item())
    if not values:raise ValueError('no valid validation outputs')
    return sum(values)/len(values),pred


def run(config, manifest, output, device='cpu', initial=None, resume=None, imagenet=None):
    config=dict(config);stage=int(config['stage'])
    if stage not in (1,2) or config['max_updates'] < 1 or min(config['validation_interval'],config['save_interval']) < 1:
        raise ValueError('invalid stage or update limit')
    if stage==2 and initial is None and resume is None:raise ValueError('Stage 2 requires Stage 1 initialization')
    if stage==1 and imagenet is None and resume is None:raise ValueError('supply verified ResNet-34 ImageNet weights for optical initialization')
    if imagenet and config.get('imagenet_sha256') and config['imagenet_sha256']!=sha256(imagenet):
        raise ValueError('ImageNet weight SHA256 differs from the configured identity')
    data=Dataset(manifest)
    data.rows=development_split(data.rows,config['validation_fraction'],config['split_seed'])
    data.by_id={r['patient_id']:r for r in data.rows}
    train=data.role('train',True);val=data.role('validation',True)
    if not train or not val:raise ValueError('training and validation patients required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'recovery.pt').exists() and resume is None:raise ValueError('existing run requires explicit resume')
    state=torch.load(resume,map_location='cpu',weights_only=False) if resume else None
    identity=dict(config=config,manifest_sha256=data.sha,**input_weight_identity(initial,imagenet,state),
                  role_hashes={role:sorted(hashlib.sha256(r['patient_id'].encode()).hexdigest() for r in data.role(role)) for role in ('train','validation','test')})
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(int(config.get('cpu_threads',2)))
    seed_all(config['seed'])
    model=Model(imagenet_path=imagenet,**config['model']).to(device)
    if initial:
        saved=torch.load(initial,map_location='cpu',weights_only=False)
        if saved['stage']!=1:raise ValueError('expected a Stage 1 checkpoint')
        model.load_state_dict(saved['model'],strict=True)
        previous=saved['identity'].get('role_hashes')
        if previous is None:raise ValueError('initial checkpoint requires patient-role identities')
        active=identity['role_hashes']
        if (set(previous['test']) & (set(active['train'])|set(active['validation'])) or
            set(active['test']) & (set(previous['train'])|set(previous['validation']))):
            raise ValueError('test patient identity overlaps model development across stages')
        if set(previous['train']) & set(active['validation']):
            raise ValueError('validation patients were fitted during the preceding stage')
    opt=torch.optim.AdamW(parameter_groups(model,stage),weight_decay=1e-4)
    sampler=Sampler(train,stage,config['seed']);aug=random.Random(config['seed']+1)
    step=0;best=float('inf');bad=0
    if resume:
        if state['identity_sha256']!=identity_sha:raise ValueError('recovery identity mismatch')
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        sampler.load_state_dict(state['sampler']);aug.setstate(state['augmentation_rng']);restore_rng(state['rng'])
        step=state['step'];best=state['best_validation'];bad=state['bad_validations']
    write_json(output/'run_identity.json',dict(**identity,identity_sha256=identity_sha))
    write_json(output/'patient_split.json',{'validation_fraction':config['validation_fraction'],'split_seed':config['split_seed'],
               'patients':[{'patient_id':r['patient_id'],'split':r['split']} for r in data.rows]})
    def snapshot():
        return dict(model=model.state_dict(),model_config=model.model_config,stage=stage,step=step,
                    optimizer=opt.state_dict(),sampler=sampler.state_dict(),augmentation_rng=aug.getstate(),rng=rng_state(),
                    identity=identity,identity_sha256=identity_sha,best_validation=best,bad_validations=bad)
    if step==0:atomic_save(output/'step_0000.pt',snapshot())
    while step<config['max_updates']:
        model.train();opt.zero_grad(set_to_none=True);ids=sampler.batch();loss_value=0.;used=0
        for pid in ids:
            row=data.by_id[pid];s=augment(data.sample(row),aug)
            s=mask_arterial(s,aug,float(config.get('arterial_dropout',.5 if stage==1 else 0.)),stage)
            loss=patient_loss(model,s,row['label'],stage,float(row.get('weight',1.)))
            if loss is not None and loss.requires_grad:
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite patient loss')
                (loss/8).backward();loss_value+=float(loss.detach())/8;used+=1
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not used or not grads or not any(g.abs().max()>0 for g in grads):
            raise RuntimeError('batch has no effective gradient; no optimizer update performed')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();step+=1
        with (output/'updates.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(step=step,patients=ids,loss=loss_value,supervised=used,gradient_norm=float(norm)))+'\n')
        if step%config['validation_interval']==0 or step==config['max_updates']:
            score,_=validation(model,data,val,stage)
            if score<best:
                best=score;bad=0;atomic_save(output/'best.pt',snapshot())
            else:bad+=1
        if step%5==0 or step==config['max_updates']:
            old=output/'recovery.pt'
            if old.exists():old.replace(output/'previous_recovery.pt')
            atomic_save(old,snapshot())
        if step%config['save_interval']==0 or step==config['max_updates']:
            atomic_save(output/f'step_{step:04d}.pt',snapshot())
        if config.get('patience',0) and bad>=config['patience']:break
    atomic_save(output/'final.pt',snapshot())
    state=torch.load(output/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    _,pred=validation(model,data,val,stage)
    thresholds={}
    for branch in ('ct','optical','joint'):
        selected=[r for r in pred if r['outputs']['all'].get(branch) is not None and
                  (branch!='joint' or r['outputs']['all'].get('joint_route')=='joint')]
        y=[data.by_id[r['patient_id']]['label'] for r in selected]
        thresholds[branch]=youden(y,[r['outputs']['all'][branch]['probability'] for r in selected],'validation') if len(set(y))==2 else None
    write_json(output/'thresholds.json',dict(checkpoint_sha256=sha256(output/'best.pt'),manifest_sha256=data.sha,
               role='validation',stage=stage,thresholds=thresholds,patient_ids=[r['patient_id'] for r in val]))
    write_json(output/'validation_predictions.json',pred)
    write_json(output/'COMPLETE.json',dict(actual_updates=step,best_step=state['step'],checkpoint_sha256=sha256(output/'best.pt')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--initial')
    p.add_argument('--resume');p.add_argument('--imagenet')
    a=p.parse_args();config=json.loads(Path(a.config).read_text(encoding='utf-8'))
    run(config,a.manifest,a.output,a.device,a.initial,a.resume,a.imagenet)


if __name__=='__main__':main()
