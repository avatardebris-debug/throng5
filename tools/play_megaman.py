"""
play_megaman.py — Human-play verification for Mega Man 2.

Loads from the saved start state, opens a window, lets you play.
Shows live reward, RAM values, and x-position so you can verify
the reward function is working correctly.

Controls: Arrow keys, Z=Jump, X=Shoot, Enter=Start
Press R to reset to saved state.
Press D to dump current RAM addresses (for debugging reward).
Press Q to quit.
"""

import sys, os, pickle, time
# Add throng5 root to path (tools/ is one level down)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pygame

ROM_PATH   = os.path.join(ROOT, "roms", "nes", "Mega Man 2 (USA)_ines1.nes")
STATE_PATH = os.path.join(ROOT, "roms", "nes", "megaman2_stage_start.pkl")
SCALE      = 3
W, H       = 256 * SCALE, 240 * SCALE + 80   # extra 80px for HUD

pygame.init()


def keys_to_bitmask(keys):
    action = 0
    if keys[pygame.K_z]:      action |= 0x01  # A = Jump
    if keys[pygame.K_x]:      action |= 0x02  # B = Shoot
    if keys[pygame.K_RSHIFT]: action |= 0x04  # Select
    if keys[pygame.K_RETURN]: action |= 0x08  # Start
    if keys[pygame.K_UP]:     action |= 0x10
    if keys[pygame.K_DOWN]:   action |= 0x20
    if keys[pygame.K_LEFT]:   action |= 0x40
    if keys[pygame.K_RIGHT]:  action |= 0x80
    return action


def main():
    import nes_py

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MM2 Verifier — R=Reset  D=Dump RAM  Q=Quit")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)

    env = nes_py.NESEnv(ROM_PATH)
    obs = env.reset()

    # Load saved state
    start_ram = None
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "rb") as f:
            data = pickle.load(f)
        start_ram = data.get("ram")
        print(f"Loaded state from step {data.get('step')}")
    else:
        print("WARNING: No saved state — starting from title screen")

    def load_state():
        nonlocal obs
        obs = env.reset()
        if start_ram is not None:
            try:
                env.ram[:] = start_ram
            except Exception as e:
                print(f"RAM restore error: {e}")
        return obs

    load_state()

    total_r   = 0.0
    step      = 0
    prev_x    = 0
    history   = []   # (step, x, reward)

    print("\nControls: Arrow=Move  Z=Jump  X=Shoot  Enter=Start")
    print("          R=Reset   D=Dump RAM values   Q=Quit\n")

    # RAM addresses to watch — we'll scan for x-position
    # Try a few candidates and show them all
    X_CANDIDATES = {
        "ram[0x00]": 0x00,
        "ram[0x04]": 0x04,
        "ram[0x20]": 0x20,
        "ram[0x46]": 0x46,   # MM2 player x screen pos (candidate)
        "ram[0x64]": 0x64,
        "ram[0x80]": 0x80,
        "ram[0xAD]": 0xAD,
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs = load_state()
                    total_r = 0.0
                    step = 0
                    prev_x = 0
                    history.clear()
                    print("Reset to saved state.")
                if event.key == pygame.K_d:
                    print(f"\n─── RAM Dump at step {step} ───")
                    for name, addr in X_CANDIDATES.items():
                        print(f"  {name} = {int(env.ram[addr]):3d}  (0x{int(env.ram[addr]):02X})")
                    print(f"  full ram[0x00:0x10] = {list(env.ram[0x00:0x10])}")
                    print(f"  ram[0x40:0x50]      = {list(env.ram[0x40:0x50])}")
                    print(f"  ram[0x60:0x70]      = {list(env.ram[0x60:0x70])}")

        keys = pygame.key.get_pressed()
        bitmask = keys_to_bitmask(keys)

        obs, _, done, info = env.step(bitmask)
        step += 1

        # Compute x-delta across candidates
        x_vals = {name: int(env.ram[addr]) for name, addr in X_CANDIDATES.items()}

        # Use candidate 0x46 as primary x guess
        curr_x = x_vals["ram[0x46]"]
        dx = curr_x - prev_x
        if dx > 128: dx -= 256
        if dx < -128: dx += 256
        r = dx * 0.01
        total_r += r
        prev_x = curr_x

        history.append((step, curr_x, r))

        if done:
            obs = load_state()
            print(f"Episode done at step {step}, total_r={total_r:.2f}")
            total_r = 0.0
            step    = 0
            prev_x  = 0

        # ── Render ─────────────────────────────────────────────────────
        # Game frame
        surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (256 * SCALE, 240 * SCALE))
        screen.blit(surf, (0, 0))

        # HUD below game frame
        hud_y = 240 * SCALE + 4
        screen.fill((20, 20, 30), (0, 240 * SCALE, W, 80))

        lines = [
            f"Step:{step:5d}  TotalR:{total_r:7.3f}  dx:{dx:+3d}",
            f"x[0x46]={curr_x:3d}  x[0x04]={x_vals['ram[0x04]']:3d}  "
            f"x[0x20]={x_vals['ram[0x20]']:3d}  x[0x80]={x_vals['ram[0x80]']:3d}",
            f"R=Reset  D=DumpRAM  Q=Quit  Z=Jump  X=Shoot  Arrows=Move",
        ]
        colors = [(255,255,100), (100,255,200), (150,150,150)]
        for i, (line, col) in enumerate(zip(lines, colors)):
            lbl = font.render(line, True, col)
            screen.blit(lbl, (6, hud_y + i * 18))

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()
    print(f"\nFinal: steps={step}  total_reward={total_r:.3f}")

    # Print which RAM address showed the most movement (= likely x-pos)
    if history:
        print("\n─── Which RAM address tracked x well? (press D while playing) ───")
        print("Run again and press D at different x positions to find the right addr.")


if __name__ == "__main__":
    main()
