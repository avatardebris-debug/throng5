"""
record_start_sequence.py — Record button inputs to navigate from title
to stage start, then save them for automatic replay in training.

Instead of saving RAM (which doesn't restore emulator state), we save
the EXACT sequence of button presses that navigates the menus.
Each episode replays this sequence at high speed, then hands off to agent.

Controls: same as always (Z=Jump, X=Shoot, Arrows=Move, Enter=Start)
Press S to mark the stage start point and save the sequence.
Press Q to quit.
"""

import sys, os, pickle, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pygame

ROM_PATH   = os.path.join(ROOT, "roms", "nes", "Mega Man 2 (USA)_ines1.nes")
SEQ_PATH   = os.path.join(ROOT, "roms", "nes", "megaman2_input_sequence.pkl")

pygame.init()
SCALE = 3
W, H  = 256 * SCALE, 240 * SCALE + 50

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
    pygame.display.set_caption(
        "Record Menu Sequence — Navigate to stage start, press S to save"
    )
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)

    env = nes_py.NESEnv(ROM_PATH)
    obs = env.reset()

    sequence = []   # list of (bitmask, obs_frame_index)
    saved    = False
    step     = 0

    print("=" * 60)
    print("RECORD MENU SEQUENCE")
    print("=" * 60)
    print("Play through ALL menus to reach stage entry point.")
    print("At the moment Mega Man can move — press S to save.")
    print("This sequence will be auto-replayed each training episode.")
    print("=" * 60)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close(); pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    env.close(); pygame.quit(); return
                if event.key == pygame.K_s:
                    data = {
                        "sequence": sequence,
                        "n_frames": len(sequence),
                        "rom": os.path.basename(ROM_PATH),
                    }
                    with open(SEQ_PATH, "wb") as f:
                        pickle.dump(data, f)
                    print(f"\nSaved {len(sequence)} frames → {SEQ_PATH}")
                    saved = True

        keys = pygame.key.get_pressed()
        bitmask = keys_to_bitmask(keys)
        sequence.append(bitmask)

        obs, _, done, info = env.step(bitmask)
        step += 1

        if done:
            print("Game over/reset — restarting recording")
            env.reset()
            sequence.clear()
            step = 0
            saved = False

        # Render
        surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (256 * SCALE, 240 * SCALE))
        screen.blit(surf, (0, 0))

        hud_y = 240 * SCALE + 4
        screen.fill((20, 20, 30), (0, 240 * SCALE, W, 50))
        status = f"Frame:{step:5d}  {'[SAVED — press Q to quit]' if saved else 'S=Save at stage start  Q=Quit'}"
        lbl = font.render(status, True, (255, 255, 100))
        screen.blit(lbl, (6, hud_y))
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
