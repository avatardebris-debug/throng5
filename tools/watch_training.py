"""
watch_training.py — Live viewer for train_megaman.py

Run in a separate terminal while training is running:
    python tools/watch_training.py

Training writes NES frames to shared memory every few steps.
This script reads them and displays at 30fps without blocking training.
"""

import sys, os, time, struct
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pygame
from multiprocessing.shared_memory import SharedMemory

SHM_NAME  = "throng5_mm2_frame"
FRAME_W, FRAME_H = 256, 240
FRAME_BYTES = FRAME_W * FRAME_H * 3
SCALE = 3

def main():
    pygame.init()
    screen = pygame.display.set_mode((FRAME_W * SCALE, FRAME_H * SCALE + 40))
    pygame.display.set_caption("Throng5 x MM2 — Live Viewer")
    clock = pygame.font.SysFont("monospace", 13)
    font  = clock
    clk   = pygame.time.Clock()

    shm = None
    waiting_printed = False

    print("Waiting for training to start (needs --render flag)...")
    print("Run: python train_megaman.py --render")

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                if shm: shm.close()
                pygame.quit()
                return

        # Try to connect to shared memory
        if shm is None:
            try:
                shm = SharedMemory(name=SHM_NAME, create=False)
                print("Connected to training frame buffer!")
            except Exception:
                # Not started yet — show waiting screen
                screen.fill((15, 15, 25))
                lbl = font.render("Waiting for training... (python train_megaman.py --render)", True, (200, 200, 100))
                screen.blit(lbl, (10, FRAME_H * SCALE // 2))
                pygame.display.flip()
                clk.tick(10)
                continue

        # Read frame from shared memory
        try:
            raw = np.frombuffer(shm.buf, dtype=np.uint8).reshape((FRAME_H, FRAME_W, 3)).copy()
            surf = pygame.surfarray.make_surface(raw.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (FRAME_W * SCALE, FRAME_H * SCALE))
            screen.blit(surf, (0, 0))

            # Status bar
            screen.fill((15, 15, 25), (0, FRAME_H * SCALE, FRAME_W * SCALE, 40))
            t = font.render(f"Live training viewer  |  {time.strftime('%H:%M:%S')}  |  Q=Quit", True, (100, 255, 100))
            screen.blit(t, (6, FRAME_H * SCALE + 10))
            pygame.display.flip()
        except Exception as e:
            pass

        clk.tick(30)

if __name__ == "__main__":
    main()
