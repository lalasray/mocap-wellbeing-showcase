"""C3D measurements in metres; missing observations remain missing."""
from pathlib import Path
import csv
import ezc3d
import numpy as np


def derivative(values, rate, half_window):
    """Symmetric finite difference; reject any missing point in its interval."""
    h = half_window
    result = np.full_like(values, np.nan)
    if len(values) <= 2 * h:
        return result
    d = (values[2*h:] - values[:-2*h]) * rate / (2*h)
    valid = np.isfinite(values).all(axis=1)
    counts = np.convolve(valid.astype(int), np.ones(2*h+1, dtype=int), mode='valid')
    d[counts != 2*h+1] = np.nan
    result[h:-h] = d
    return result


class Recording:
    def __init__(self, path):
        self.path = Path(path)
        c = ezc3d.c3d(str(path))
        params = c['parameters']['POINT']
        self.rate = float(params['RATE']['value'][0])
        units = params['UNITS']['value'][0].strip().lower()
        if units not in {'m', 'mm', 'cm'}:
            raise ValueError(f'Unsupported coordinate units: {units!r}')
        scale = {'m': 1., 'mm': .001, 'cm': .01}[units]
        labels = params['LABELS']['value']
        keep = [i for i, name in enumerate(labels) if ':' in name and not name.lower().startswith('unlabeled')]
        self.labels = [labels[i].split(':')[-1] for i in keep]
        if len(set(self.labels)) != len(self.labels):
            raise ValueError('Multiple subjects detected. Export one subject for this demo.')
        if not keep:
            raise ValueError('No subject-labelled markers found. This demo expects the supplied full-body marker set.')
        self.xyz = c['data']['points'][:3, keep, :].transpose(2, 1, 0) * scale
        residuals = c['data']['meta_points']['residuals'][0, keep, :].T
        self.xyz[(residuals < 0) | ~np.isfinite(self.xyz).all(axis=2)] = np.nan
        self.valid = np.isfinite(self.xyz).all(axis=2)
        self.time = np.arange(len(self.xyz)) / self.rate
        if len(self.time) < 3 or self.rate <= 0:
            raise ValueError('Recording must contain at least three frames at a positive rate.')
        self.coverage = self.valid.mean(axis=0)
        self.lookup = {label: i for i, label in enumerate(self.labels)}
        self.pelvis = np.mean([self.marker(n) for n in
            ['WaistLFront', 'WaistLBack', 'WaistRFront', 'WaistRBack']], axis=0)
        # Approximately 100 ms finite difference suppresses frame-to-frame jitter.
        self.half_window = max(1, round(self.rate * .05))
        velocity = derivative(self.pelvis, self.rate, self.half_window)
        self.speed = np.linalg.norm(velocity[:, :2], axis=1)
        wrist_speeds, angles = [], []
        torso = self.marker('ChestTop') - self.pelvis
        trunk = (self.marker('ChestTop') + self.marker('BackTop')) / 2 - self.pelvis
        with np.errstate(invalid='ignore', divide='ignore'):
            self.trunk_lean = np.degrees(np.arccos(np.clip(
                trunk[:,2] / np.linalg.norm(trunk,axis=1), -1, 1)))
        for side in 'LR':
            wrist = (self.marker(side+'WristOut') + self.marker(side+'WristIn')) / 2
            wrist_speeds.append(np.linalg.norm(derivative(wrist-self.pelvis, self.rate, self.half_window), axis=1))
            arm = self.marker(side+'ElbowOut') - self.marker(side+'ShoulderTop')
            denom = np.linalg.norm(arm, axis=1) * np.linalg.norm(torso, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                cosine = np.sum(arm * -torso, axis=1) / denom
                angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        self.wrist_speed = np.mean(wrist_speeds, axis=0)
        self.angles = np.array(angles).T

    def marker(self, name):
        if name not in self.lookup:
            raise ValueError(f'Required marker absent: {name}')
        return self.xyz[:, self.lookup[name]]

    def summary(self, start, end):
        if end <= start:
            raise ValueError('Interval end must be after start.')
        mask = (self.time >= start) & (self.time <= end)
        if not mask.any():
            raise ValueError('Interval contains no frames.')
        def mean(x):
            return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')
        ranges = []
        for a in self.angles[mask].T:
            a = a[np.isfinite(a)]
            ranges.append(float(np.ptp(np.percentile(a, [5, 95]))) if len(a) > 1 else float('nan'))
        return {'Walking speed (m/s)': mean(self.speed[mask]),
                'Relative wrist speed (m/s)': mean(self.wrist_speed[mask]),
                'Left arm range (deg)': ranges[0], 'Right arm range (deg)': ranges[1],
                'Valid named markers (%)': float(self.valid[mask].mean()*100),
                'Valid speed samples (%)': float(np.isfinite(self.speed[mask]).mean()*100),
                'Valid wrist samples (%)': float(np.isfinite(self.wrist_speed[mask]).mean()*100)}

    def rolling_summary(self, index, seconds):
        """Trailing window, clipped to the start; never wrap into the previous loop."""
        if not 3 <= seconds <= 15:
            raise ValueError('Rolling window must be between 3 and 15 seconds.')
        if not 0 <= index < len(self.time):
            raise ValueError('Frame index outside recording.')
        end = float(self.time[index])
        start = max(0., end-seconds)
        summary = self.summary(start, max(end, np.finfo(float).eps))
        return start, end, summary

    def export(self, folder):
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        with (folder/'measurements.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s','pelvis_horizontal_speed_m_s','pelvis_relative_wrist_speed_m_s',
                             'left_arm_to_downward_trunk_deg','right_arm_to_downward_trunk_deg','valid_marker_percent'])
            writer.writerows(zip(self.time, self.speed, self.wrist_speed, *self.angles.T, self.valid.mean(axis=1)*100))
