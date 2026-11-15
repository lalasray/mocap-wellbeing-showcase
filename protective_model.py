"""Pose subhead of the released EmoPain-Fusion BI-BA-CNN checkpoint.

This is a cross-sensor transfer, not the published pose+EMG ensemble.
No synthetic EMG, retraining, or emotion-to-pain conversion is performed.
"""
from pathlib import Path
import hashlib
import numpy as np
from emotion_model import fill_short_gaps

BASE = Path(__file__).resolve().parent
SHA256 = '916cf28e65042910baa7c41e96c5a27073446f78a87de0b877450589fc2800f7'
COMMIT = '219e88c02ed65daaa4cd772d53745204e0b07048'
URL = f'https://raw.githubusercontent.com/981526092/EmoPain-Fusion/{COMMIT}/Best_Model_Weights/HOT_bi_cnn_banet_coordinate.hdf5'
CHECKPOINT = BASE/'models/HOT_bi_cnn_banet_coordinate.hdf5'
LABELS = ['Non-protective', 'Protective']


def map_markers(recording):
    """Approximate the 26-node EmoPain skeleton minus heels/toes (22 nodes).

    Source order: hip, left hip/knee/ankle, right hip/knee/ankle,
    spine/lower spine, left clavicle/shoulder/elbow/wrist/fingertip,
    right equivalents, neck/head/crown. Surface markers are proxies.
    Convert metres Z-up to centimetres Y-up with a proper rotation.
    Retain world translation, as the release uses raw coordinates.
    """
    m = recording.marker
    avg = lambda *names: np.mean([m(n) for n in names], axis=0)
    pelvis = recording.pelvis
    neck = avg('ChestTop', 'BackTop')
    left, right = m('LShoulderTop'), m('RShoulderTop')
    nodes = [pelvis, avg('WaistLFront','WaistLBack'), m('LKneeOut'), m('LHeel'),
             avg('WaistRFront','WaistRBack'), m('RKneeOut'), m('RHeel'),
             pelvis+(neck-pelvis)/3, pelvis+2*(neck-pelvis)/3,
             (neck+left)/2, left, m('LElbowOut'), avg('LWristIn','LWristOut'),
             avg('LHandIn','LHandOut'), (neck+right)/2, right, m('RElbowOut'),
             avg('RWristIn','RWristOut'), avg('RHandIn','RHandOut'), neck,
             avg('HeadFront','HeadLeft','HeadRight'), m('HeadTop')]
    p = np.stack(nodes, axis=1)
    return p[:,:,[0,2,1]] * np.array([100.,100.,-100.])


class ProtectiveModel:
    def __init__(self, path=CHECKPOINT):
        import h5py
        import torch
        self.torch = torch
        if not path.exists():
            raise FileNotFoundError('Run setup_protective_model.py.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Protective checkpoint checksum mismatch.')
        self.weights = {}
        with h5py.File(path, 'r') as f:
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    self.weights[name] = torch.tensor(obj[()], dtype=torch.float32)
            f.visititems(collect)

    def infer(self, windows):
        """Exact inference equations for suboutput1; input (N,180,22,3)."""
        t = self.torch
        w = self.weights
        if windows.ndim != 4 or windows.shape[1:] != (180,22,3):
            raise ValueError('Expected (N,180,22,3) coordinates.')
        if not np.isfinite(windows).all():
            raise ValueError('Nonfinite pose input.')
        def conv(x, prefix):
            return t.nn.functional.conv1d(x, w[prefix+'/kernel:0'].permute(2,1,0), w[prefix+'/bias:0'])
        def bn(x, prefix):
            return ((x-w[prefix+'/moving_mean:0'][None,:,None]) /
                    t.sqrt(w[prefix+'/moving_variance:0'][None,:,None]+.001) *
                    w[prefix+'/gamma:0'][None,:,None]+w[prefix+'/beta:0'][None,:,None])
        def dense(x, name):
            return x@w[f'{name}/{name}/kernel:0']+w[f'{name}/{name}/bias:0']
        with t.inference_mode():
            x = t.as_tensor(windows, dtype=t.float32).permute(0,2,3,1).reshape(-1,3,180)
            x = t.relu(bn(conv(x,'model_15/conv1d_140'),'model_15/batch_normalization_20'))
            x = t.tanh(bn(conv(x,'model_15/conv1d_141'),'model_15/batch_normalization_21'))
            x = x.reshape(len(windows),22,30,-1)
            parts = []
            for j in range(22):
                name = f'conv1d_{146+j}'
                attention = t.softmax(conv(x[:,j],f'{name}/{name}'),dim=-1)
                parts.append((x[:,j]*attention).sum(dim=-1))
            data = t.stack(parts,dim=-1)  # N,30,22; Keras Flatten order
            attention = t.softmax(dense(t.tanh(dense(data,'dense_10')),'bodyattention'),dim=-1)
            return t.softmax(dense((data*attention).flatten(1),'suboutput1'),dim=-1).numpy()

    def predict(self, recording, mapped, index, seconds):
        end = float(recording.time[index])
        result = dict(start_s=max(0.,end-seconds), end_s=end, window_seconds=seconds,
                      scores=None, status='', checkpoint_sha256=SHA256,
                      model='EmoPain-Fusion BI-BA-CNN pose suboutput1')
        if end < seconds:
            result['status'] = f'Waiting for a full {seconds}-second window.'
            return result
        start = np.searchsorted(recording.time,end-seconds)
        raw = mapped[start:index+1]
        try:
            coverage = float(np.isfinite(raw).all(axis=2).mean())
            result['joint_coverage'] = coverage
            if coverage < .98:
                raise ValueError('Less than 98% of required joint samples available.')
            raw = fill_short_gaps(raw,recording.rate)
            # Preserve speed: native 3 s / 180 frames at 60 Hz. Average
            # overlapping 3-second predictions across the selected interval.
            offsets = np.linspace(0,seconds-3,max(1,round(seconds-3)+1))
            times = recording.time[start:index+1]
            flat = raw.reshape(len(raw),-1)
            windows = []
            for offset in offsets:
                target = end-seconds+offset+np.arange(180)/60
                windows.append(np.stack([np.interp(target,times,c) for c in flat.T],axis=-1).reshape(180,22,3))
            scores = self.infer(np.asarray(windows)).mean(axis=0)
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite model output.')
            result.update(scores=dict(zip(LABELS,map(float,scores))), subwindows=len(windows))
        except ValueError as error:
            result['status'] = str(error)
        return result
