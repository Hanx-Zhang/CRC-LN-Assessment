"""Geometry checks and packaging of candidate inputs for the diagnostic model."""
import numpy as np
from PIL import Image


def verify_geometry(image_geometry, label_geometry):
    for key in ('shape','spacing','origin','direction'):
        if not np.array_equal(image_geometry[key],label_geometry[key]):
            raise ValueError('image/label geometry differs: '+key)


def intensity_summary(volume):
    x=np.asarray(volume)
    if not x.size or not np.isfinite(x).all():raise ValueError('empty or nonfinite CT')
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std()))


def optical_crop(image, center, side=128):
    """Crop on the documented standard canvas; pad beyond canvas with zero."""
    x,y=center;left=int(round(x-side/2));top=int(round(y-side/2))
    crop=image.crop((left,top,left+side,top+side))
    return crop,dict(canvas_size=list(image.size),crop_xyxy=[left,top,left+side,top+side],encoder_size=[224,224])


def pack_ct(path,normalized_ct,candidate_mask,uids,station):
    ct=np.asarray(normalized_ct,dtype=np.float32);mask=np.asarray(candidate_mask)
    if ct.shape!=mask.shape or ct.shape[1:]!=(32,32,32) or not np.isfinite(ct).all():
        raise ValueError('aligned normalized N x 32 x 32 x 32 candidate inputs required')
    if not np.isin(mask,[0,1]).all() or len(uids)!=len(ct) or len(set(uids))!=len(uids) or len(station)!=len(ct):
        raise ValueError('invalid candidate mask or identities')
    np.savez_compressed(path,ct=ct,target_mask=mask.astype(np.uint8),uids=np.asarray(uids,dtype=str),station=np.asarray(station,dtype=np.int64))
