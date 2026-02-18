import sys, os, numpy as np
sys.path.insert(0, r'C:\Users\avata\aicompete\throng3')
from throng4.basal_ganglia.dreamer_engine import WorldModel, NetworkSize, NETWORK_CONFIGS
from throng4.basal_ganglia.compressed_state import CompressedStateEncoder, EncodingMode

model = WorldModel(64, 4, NETWORK_CONFIGS[NetworkSize.MICRO])

def gt(s, a):
    ns = np.roll(s, a+1)*0.95
    ns[a*16] += 0.5
    return ns, float(np.sum(ns[:4])*0.1)

for i in range(500):
    s = np.random.randn(64).astype(np.float32)*0.5
    a = np.random.randint(4)
    s2, r = gt(s, a)
    model.update(s, a, s2.astype(np.float32), r, lr=0.001)

print("CALIBRATION DRIFT:")
for depth in [1, 5, 10, 20, 30, 60, 120]:
    errs = []
    for _ in range(30):
        ts = np.random.randn(64).astype(np.float32)*0.5
        ps = ts.copy()
        for step in range(depth):
            a = np.random.randint(4)
            ts, _ = gt(ts, a)
            ps, _ = model.predict(ps, a)
            ts = ts.astype(np.float32)
            ps = ps.astype(np.float32)
        errs.append(np.mean(np.abs(ts-ps)))
    avg = np.mean(errs)
    status = "DRIFT" if avg > 1.0 else "ok"
    print(f"d={depth} err={avg:.4f} {status}")

print()
print("COMPRESSED VS UNCOMPRESSED:")
for n_lev in [2, 4, 8, 0]:
    m = WorldModel(64, 4, NETWORK_CONFIGS[NetworkSize.MICRO])
    enc = None
    if n_lev > 0:
        enc = CompressedStateEncoder(
            mode=EncodingMode.QUANTIZED,
            n_quantize_levels=n_lev,
        )
    for i in range(500):
        s = np.random.randn(64).astype(np.float32)*0.5
        a = np.random.randint(4)
        s2, r = gt(s, a)
        if enc:
            cs = enc.encode(s).data
            cs2 = enc.encode(s2.astype(np.float32)).data
            m.update(cs, a, cs2, r, lr=0.001)
        else:
            m.update(s, a, s2.astype(np.float32), r, lr=0.001)
    e1 = []
    e10 = []
    for _ in range(50):
        s = np.random.randn(64).astype(np.float32)*0.5
        a = np.random.randint(4)
        tn, _ = gt(s, a)
        cs = enc.encode(s).data if enc else s
        pn, _ = m.predict(cs, a)
        e1.append(np.mean(np.abs(tn-pn)))
        ps = cs.copy()
        ts2 = s.copy()
        for step in range(10):
            a = np.random.randint(4)
            ts2, _ = gt(ts2, a)
            ps, _ = m.predict(ps, a)
        e10.append(np.mean(np.abs(ts2-ps)))
    if n_lev > 0:
        label = f"{n_lev}-level"
    else:
        label = "raw"
    print(f"{label:>8s}: 1-step={np.mean(e1):.4f}  10-step={np.mean(e10):.4f}")
