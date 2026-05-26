"""
auto_navigate.py — Automatically navigate MM2 menus and record the sequence.

No user input needed to get through menus — scripted frame-perfect timing.
Opens a window so you can SEE it working.
At stage start, auto-saves the sequence and then hands control to YOU for
5 seconds so you can verify Mega Man responds to controls.

Run once, then use train_megaman.py normally.
"""

import sys, os, pickle, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pygame

ROM_PATH = os.path.join(ROOT, "roms", "nes", "Mega Man 2 (USA)_ines1.nes")
SEQ_PATH = os.path.join(ROOT, "roms", "nes", "megaman2_input_sequence.pkl")  # human recording wins
AUTO_PATH = os.path.join(ROOT, "roms", "nes", "megaman2_auto_sequence.pkl")  # auto saves here

# NES bitmasks
NOOP   = 0x00
A      = 0x01   # Jump
B      = 0x02   # Shoot
START  = 0x08
UP     = 0x10
DOWN   = 0x20
LEFT   = 0x40
RIGHT  = 0x80

def keys_to_bitmask(keys):
    action = 0
    if keys[pygame.K_z]:      action |= A
    if keys[pygame.K_x]:      action |= B
    if keys[pygame.K_RSHIFT]: action |= 0x04
    if keys[pygame.K_RETURN]: action |= START
    if keys[pygame.K_UP]:     action |= UP
    if keys[pygame.K_DOWN]:   action |= DOWN
    if keys[pygame.K_LEFT]:   action |= LEFT
    if keys[pygame.K_RIGHT]:  action |= RIGHT
    return action

# ── Scripted menu navigation ─────────────────────────────────────────────
# Each entry: (start_frame, end_frame, bitmask)
# Ranges where we hold the given button(s)
SCRIPT = [
    # ── Title screen → Start/Password select ──────────────────────────
    (180, 185, START),   # Press Start on title screen
    (185, 230, NOOP),    # Wait for Start/Password menu

    # ── Start/Password → Difficulty select ────────────────────────────
    (230, 235, START),   # Press Start (choose "Start", not Password)
    (235, 290, NOOP),    # Wait for difficulty menu

    # ── Difficulty → Stage select (Normal) ────────────────────────────
    (290, 295, START),   # Press Start for Normal difficulty
    (295, 340, NOOP),    # Wait for stage select screen

    # ── Stage select: cursor starts at Dr. Wily (center) ──────────────
    # UP moves cursor to Air Man (top center), then START to select
    (420, 430, UP),      # Move cursor up → Air Man (wait longer for screen to load)
    (430, 470, NOOP),    # Brief pause
    (470, 475, START),   # Select Air Man
    (475, 900, NOOP),    # Wait for Air Man stage to fully load
]

RECORD_UNTIL  = 900
VERIFY_FRAMES = 300


def get_scripted_action(frame: int) -> int:
    for (s, e, btn) in SCRIPT:
        if s <= frame < e:
            return btn
    return NOOP


def main():
    import nes_py

    pygame.init()
    SCALE = 3
    W, H  = 256 * SCALE, 240 * SCALE + 60
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MM2 Auto-Navigate — verifying controls after stage load")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)

    env = nes_py.NESEnv(ROM_PATH)
    obs = env.reset()

    sequence   = []
    frame      = 0
    phase      = "auto"   # "auto" → "verify" → "done"
    saved      = False

    print("Auto-navigating Mega Man 2 menus...")
    print(f"Recording {RECORD_UNTIL} frames, then {VERIFY_FRAMES} frames for manual verification.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close(); pygame.quit(); return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                env.close(); pygame.quit(); return

        keys = pygame.key.get_pressed()

        if phase == "auto":
            action = get_scripted_action(frame)
            sequence.append(action)

            if frame == RECORD_UNTIL - 1:
                # Save sequence
                data = {"sequence": sequence, "n_frames": len(sequence)}
                with open(SEQ_PATH, "wb") as f:
                    pickle.dump(data, f)
                print(f"\n✓ Saved {len(sequence)}-frame sequence → {SEQ_PATH}")
                print("Now take control — verify Mega Man responds to your controls!")
                print("Controls: Arrow keys, Z=Jump, X=Shoot")
                saved = True
                phase = "verify"

        elif phase == "verify":
            action = keys_to_bitmask(keys)
            if frame >= RECORD_UNTIL + VERIFY_FRAMES:
                phase = "done"
                print("\nDone! Sequence saved. Run: python train_megaman.py")

        else:  # done
            action = keys_to_bitmask(keys)

        obs, _, done, info = env.step(action)
        frame += 1

        if done and phase == "auto":
            print(f"  Game reset at frame {frame} — restarting")
            obs = env.reset()
            sequence.clear()
            frame = 0
            saved = False

        # Render
        surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (256 * SCALE, 240 * SCALE))
        screen.blit(surf, (0, 0))

        # HUD
        hud_y = 240 * SCALE + 4
        screen.fill((20, 20, 30), (0, 240 * SCALE, W, 60))

        if phase == "auto":
            action_name = {NOOP:"NOOP", START:"START", A:"A(Jump)", B:"B(Shoot)"}.get(action, hex(action))
            line1 = f"AUTO | Frame:{frame:4d}/{RECORD_UNTIL}  Action:{action_name}"
            pct = (RECORD_UNTIL - frame) // 20
            line2 = f"Script progress {'░' * pct}  Q=Quit"
        elif phase == "verify":
            vf = frame - RECORD_UNTIL
            line1 = f"VERIFY | Frame:{vf:3d}/{VERIFY_FRAMES}  — Move Mega Man to verify controls!"
            line2 = f"Z=Jump  X=Shoot  Arrows=Move  Q=Quit"
        else:
            line1 = "DONE — sequence saved, close window"
            line2 = f"Run: python train_megaman.py"

        for i, (txt, col) in enumerate([(line1, (255,255,100)), (line2, (100,255,200))]):
            lbl = font.render(txt, True, col)
            screen.blit(lbl, (6, hud_y + i * 20))

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
