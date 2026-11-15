"""Experimental transfer of the published TAEW ST-GCN baseline to marker data.

Scores describe learned perceived-expression classes, not mental-health estimates.
The authors' forward pass is preserved exactly, including tensor reshaping.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
INFO = json.loads((BASE/'models/model_info.json').read_text())
LABELS = INFO['labels']


def map_markers(recording):
    """Map surface markers to 21 approximate Edinburgh joints; retain gaps."""
    m = recording.marker
    avg = lambda *names: np.mean([m(n) for n in names], axis=0)
    pelvis = recording.pelvis
    neck = avg('ChestTop','BackTop')
    nodes = [pelvis,
        avg('WaistLFront','WaistLBack'),m('LKneeOut'),m('LHeel'),m('LToeTip'),
        avg('WaistRFront','WaistRBack'),m('RKneeOut'),m('RHeel'),m('RToeTip'),
        pelvis+(neck-pelvis)/3, pelvis+2*(neck-pelvis)/3, neck,
        avg('HeadFront','HeadLeft','HeadRight','HeadTop'),
        m('LShoulderTop'),m('LElbowOut'),avg('LWristIn','LWristOut'),avg('LHandIn','LHandOut'),
        m('RShoulderTop'),m('RElbowOut'),avg('RWristIn','RWristOut'),avg('RHandIn','RHandOut')]
    return np.stack(nodes, axis=1)


def fill_short_gaps(points, rate, max_seconds=.15):
    """Fill only bounded gaps no longer than 150 ms; reject all longer gaps."""
    out = points.copy()
    for j in range(out.shape[1]):
        valid = np.isfinite(out[:,j]).all(axis=1)
        good = np.flatnonzero(valid)
        if len(good)<2 or not valid[0] or not valid[-1]:
            raise ValueError('Required joints missing at the window boundary.')
        if np.max(np.diff(good)-1) / rate > max_seconds:
            raise ValueError('Required joints missing for more than 150 ms.')
        for axis in range(3):
            out[:,j,axis] = np.interp(np.arange(len(out)),good,out[good,j,axis])
    return out


def prepare_window(joints, rate):
    """Align Y-up/+X left/+Z forward and retarget to reference bone lengths.

    One heading per window preserves turns. Time-normalize to the authors'
    48-frame inference shape; 3–15 s durations are an experimental adaptation.
    """
    p = fill_short_gaps(joints,rate)
    left = p[0,13]-p[0,17]
    left[2] = 0
    length = np.linalg.norm(left)
    if length < .05:
        raise ValueError('Shoulder geometry is degenerate.')
    left /= length
    up = np.array([0.,0.,1.])
    forward = np.cross(left,up)
    rotation = np.stack([left,up,forward],axis=1)
    p = p @ rotation
    # Relative translation in X/Z; retain vertical movement, scaled to the template.
    source_height = np.median(p[:,0,1] - (p[:,3,1]+p[:,7,1])/2)
    if source_height < .3:
        raise ValueError('Pelvis/heel geometry is unsuitable for gait inference.')
    scale = INFO['template_root_height']/source_height
    out = np.empty_like(p)
    out[:,0] = (p[:,0]-p[0,0])*scale
    out[:,0,1] += INFO['template_root_height']
    for j,parent in enumerate(INFO['parents'][1:],start=1):
        bone = p[:,j]-p[:,parent]
        norms = np.linalg.norm(bone,axis=1)
        if np.any(norms < 1e-6):
            raise ValueError('Degenerate mapped bone.')
        out[:,j] = out[:,parent]+bone/norms[:,None]*INFO['template_bone_lengths'][j]
    target = np.linspace(0,len(out)-1,INFO['frames'])
    flat = out.reshape(len(out),-1)
    sampled = np.stack([np.interp(target,np.arange(len(out)),column) for column in flat.T],axis=1)
    return sampled.reshape(1,INFO['frames'],21,3,1)


class EmotionModel:
    def __init__(self):
        import torch
        self.torch = torch
        path = BASE/'models/stgcn_500_5.pt'
        if not path.exists():
            raise FileNotFoundError('Missing pretrained weights. Run setup_model.py.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != INFO['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum mismatch. Run setup_model.py again.')
        sys.path.insert(0,str(BASE/'vendor/taew_stgcn'))
        from st_gcn import Model
        torch.set_num_threads(2)
        self.model = Model(3,4).double().eval()
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)

    def predict(self, recording, mapped, index, seconds):
        end = float(recording.time[index])
        result = {'start_s':max(0.,end-seconds),'end_s':end,'window_seconds':seconds,
                  'scores':None,'status':'','checkpoint_sha256':INFO['checkpoint_sha256']}
        if end < seconds:
            result['status'] = f'Waiting for a full {seconds}-second window.'
            return result
        start = np.searchsorted(recording.time,end-seconds)
        raw = mapped[start:index+1]
        coverage = float(np.isfinite(raw).all(axis=2).mean())
        result['joint_coverage'] = coverage
        try:
            if coverage < .98:
                raise ValueError('Less than 98% of required joint samples available.')
            speed = recording.speed[start:index+1]
            if not np.isfinite(speed).any() or np.nanmean(speed)<.15:
                raise ValueError('Too little walking motion for this gait model.')
            inputs = prepare_window(raw,recording.rate)
            with self.torch.inference_mode():
                logits = self.model(self.torch.from_numpy(inputs))
                scores = self.torch.sigmoid(logits)[0].cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError('Model returned nonfinite scores.')
            result['scores'] = dict(zip(LABELS,map(float,scores)))
            result['status'] = 'Experimental transfer • uncalibrated model scores'
        except ValueError as error:
            result['status'] = str(error)
        return result
