"""
create_start_state.py — Play Mega Man 2 manually and save a start state.

Controls:
  Arrow keys  = D-pad
  Z           = A button (Jump)
  X           = B button (Shoot)
  Enter       = Start
  Right Shift = Select
  S           = SAVE STATE and quit
  Q           = Quit without saving

Run with:
  python tools/create_start_state.py
"""

import os, sys, pickle
import numpy as np

ROM_PATH = r"C:\Users\avata\aicompete\throng5\roms\nes\Mega Man 2 (USA)_ines1.nes"
STATE_PATH = r"C:\Users\avata\aicompete\throng5\roms\nes\megaman2_stage_start.pkl"

try:
    import nes_py
    from nes_py.wrappers import JoypadSpace
    CONTROLS = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['A'],
        ['B'],
        ['left'],
        ['left', 'A'],
        ['down'],
        ['up'],
        ['START'],
        ['SELECT'],
    ]
except ImportError:
    print("ERROR: nes_py not installed. Run: pip install nes-py")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("ERROR: pygame not installed. Run: pip install pygame")
    sys.exit(1)


# nes_py bitmask order (LSB first = NES hardware read order):
#  0x01=A  0x02=B  0x04=Select  0x08=Start
#  0x10=Up 0x20=Down 0x40=Left  0x80=Right
def keys_to_action(keys) -> int:
    """Convert pygame key state to nes_py uint8 bitmask."""
    action = 0
    if keys[pygame.K_z]:      action |= 0x01  # A = Jump
    if keys[pygame.K_x]:      action |= 0x02  # B = Shoot
    if keys[pygame.K_RSHIFT]: action |= 0x04  # Select
    if keys[pygame.K_RETURN]: action |= 0x08  # Start
    if keys[pygame.K_UP]:     action |= 0x10  # Up
    if keys[pygame.K_DOWN]:   action |= 0x20  # Down
    if keys[pygame.K_LEFT]:   action |= 0x40  # Left
    if keys[pygame.K_RIGHT]:  action |= 0x80  # Right
    return action


def main():
    pygame.init()

    env = nes_py.NESEnv(ROM_PATH)
    obs = env.reset()

    # Scale up the display (NES is 256x240, show at 3x)
    SCALE = 3
    W, H = 256 * SCALE, 240 * SCALE
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        "Mega Man 2 — Navigate to stage start, press S to save, Q to quit"
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    print("\n" + "=" * 60)
    print("MEGA MAN 2 — State Creator")
    print("=" * 60)
    print("Navigate past title screen and to the stage start.")
    print("Controls:")
    print("  Arrow keys = Move    Z = Jump    X = Shoot")
    print("  Enter = Start        Right Shift = Select")
    print("  S = Save state here  Q = Quit without saving")
    print("=" * 60 + "\n")

    step = 0
    saved = False

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    if saved:
                        print("Closed (state was already saved).")
                    else:
                        print("Quit — no state saved.")
                    env.close()
                    pygame.quit()
                    return
                if event.key == pygame.K_s:
                    # Save the current emulator state
                    state = env.ram.copy()  # 2KB NES RAM snapshot
                    # Also save full env state via pickle if available
                    state_data = {
                        "ram": state,
                        "step": step,
                        "rom_path": ROM_PATH,
                        "note": "Saved via create_start_state.py",
                    }
                    with open(STATE_PATH, "wb") as f:
                        pickle.dump(state_data, f)
                    print(f"\nState saved to: {STATE_PATH}")
                    print(f"Step: {step} | RAM snapshot: {len(state)} bytes")
                    saved = True

        keys = pygame.key.get_pressed()
        action = keys_to_action(keys)
        obs, reward, done, info = env.step(action)
        step += 1

        if done:
            obs = env.reset()

        # Render: convert RGB array to pygame surface
        # obs is (240, 256, 3) uint8
        surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (W, H))
        screen.blit(surf, (0, 0))

        # HUD
        status = f"Step: {step}  {'[SAVED - press Q to quit]' if saved else 'S=Save  Q=Quit'}"
        label = font.render(status, True, (255, 255, 0), (0, 0, 0))
        screen.blit(label, (8, 8))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
