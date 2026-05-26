import pickle, nes_py, struct, os, sys

rom      = r'C:\Users\avata\aicompete\throng5\roms\nes\Mega Man 2 (USA)_ines1.nes'
seq_path = r'C:\Users\avata\aicompete\throng5\roms\nes\megaman2_input_sequence.pkl'
out      = r'C:\Users\avata\aicompete\throng5\checkpoints\ep_start_verify.bmp'
os.makedirs(r'C:\Users\avata\aicompete\throng5\checkpoints', exist_ok=True)

with open(seq_path, 'rb') as f:
    seq = pickle.load(f)['sequence']

env = nes_py.NESEnv(rom)
obs = env.reset()
aborted = False
for bitmask in seq:
    obs, _, done, _ = env.step(bitmask)
    if done:
        aborted = True
        break

print(f'Sequence replay: {"ABORTED (done fired)" if aborted else "completed OK"}')
print(f'x_EA = {int(env.ram[0xEA])}  (expected ~72 for Air Man stage start)')

raw = obs  # obs IS the pixel frame (240, 256, 3)
h, w, _ = raw.shape
row_sz, pad = w * 3, (4 - (w * 3) % 4) % 4
with open(out, 'wb') as f:
    f.write(b'BM')
    f.write(struct.pack('<I', 54 + (row_sz + pad) * h))
    f.write(b'\x00' * 4)
    f.write(struct.pack('<II', 54, 40))
    f.write(struct.pack('<ii', w, -h))
    f.write(struct.pack('<HHI', 1, 24, 0))
    f.write(struct.pack('<I', (row_sz + pad) * h))
    f.write(struct.pack('<iiII', 2835, 2835, 0, 0))
    for row in raw:
        for px in row:
            f.write(bytes([px[2], px[1], px[0]]))
        f.write(b'\x00' * pad)

env.close()
print(f'Saved: {out}')
