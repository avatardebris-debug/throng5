import sys
sys.path.insert(0, '.')
import numpy as np
from throng4.cognition.threat_estimator import ThreatEstimator

# Representative episode states for L3 and L7
cases = [
    ('bad_L3',  {'level':3, 'max_height':11, 'holes':18,  'bumpiness':25, 'lines_cleared':1,  'pieces_placed':2}),
    ('ok_L3',   {'level':3, 'max_height':4,  'holes':1,   'bumpiness':3,  'lines_cleared':40, 'pieces_placed':80}),
    ('bad_L7',  {'level':7, 'max_height':20, 'holes':95,  'bumpiness':12, 'lines_cleared':0,  'pieces_placed':1}),
    ('ok_L7',   {'level':7, 'max_height':19, 'holes':70,  'bumpiness':8,  'lines_cleared':10, 'pieces_placed':12}),
    ('great_L7',{'level':7, 'max_height':18, 'holes':55,  'bumpiness':5,  'lines_cleared':50, 'pieces_placed':25}),
]

for path in [
    'experiments/threat_estimator_L3.npz',
    'experiments/threat_estimator_L7.npz',
    'experiments/threat_estimator_all.npz',
]:
    try:
        te = ThreatEstimator.load(path)
        print(f"  {path.split('/')[-1]}  (n_trained={te.n_trained}):")
        for name, ep in cases:
            f = ThreatEstimator._episode_to_features(ep)
            if f is not None:
                p = te.predict(f)
                m = te.mode(f)
                print(f"    {name:<12} threat={p:.3f}  mode={m}")
        print()
    except Exception as e:
        print(f"  {path}: {e}")
