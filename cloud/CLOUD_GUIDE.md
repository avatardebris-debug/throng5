# Cloud Training Guide — vast.ai

## What You Need
- vast.ai account ($5-10 credit to start)
- throng5 repo on GitHub (private OK — use SSH key)
- Lolo ROM file: `Adventures of Lolo (USA).nes`

---

## Step 1: Rent a Machine

1. Go to [vast.ai/console/create](https://cloud.vast.ai/console/create/)
2. Filter: **1× GPU**, **PyTorch** image, **≥16GB RAM**
3. Recommended: RTX 3090 or 4090 (~$0.20-0.40/hr)
4. Select **SSH** access (not Jupyter — we want terminal)
5. Click **RENT** → note the SSH command they give you

## Step 2: Connect & Setup

```bash
# SSH into the machine (vast.ai gives you this command)
ssh -p PORT root@HOST

# Download and run setup
git clone https://github.com/YOUR_USERNAME/throng5.git
cd throng5
bash cloud/setup_cloud.sh
```

> **ROM**: You'll need to upload the Lolo ROM separately:
> ```bash
> # From YOUR LOCAL machine:
> scp -P PORT "Adventures of Lolo (USA).nes" root@HOST:throng5/roms/
> ```

## Step 3: Train

```bash
source .venv/bin/activate

# Run tiered GAN training (all 7 tiers, ~10-40 min)
python -u brain/games/lolo/run_tiered_gan.py 2>&1 | tee outputs/logs/train.log

# Check results
cat brain/games/lolo/gan_checkpoint.json | python -m json.tool
```

## Step 4: Export Weights

```bash
python cloud/export_weights.py
# Creates:
#   outputs/weights/sarsa_qtable.npz
#   outputs/weights/gan_generator.pt
#   outputs/weights/gan_discriminator.pt
#   outputs/weights/training_stats.json
```

## Step 5: Download to Local Machine

```bash
# FROM YOUR LOCAL Windows machine (PowerShell):
scp -P PORT root@HOST:throng5/outputs/weights/* C:\Users\avata\aicompete\throng5\outputs\weights\

# Load into local brain:
python cloud/load_weights.py outputs\weights\
```

## Step 6: NES ROM Validation (on cloud)

```bash
# Test with real Lolo ROM (Linux only — needs stable-retro)
python brain/games/lolo/test_nes_lolo.py
```

---

## Cost Estimate

| Task | GPU | Time | Cost |
|------|-----|------|------|
| Tiered GAN training | RTX 3090 | ~15 min | ~$0.05 |
| Extended training (1000+ puzzles) | RTX 3090 | ~2 hr | ~$0.50 |
| NES ROM validation | CPU only | ~10 min | ~$0.03 |
| **Total per session** | | | **~$0.50-1.00** |

## Quick Reference

```bash
# Connect
ssh -p PORT root@HOST

# Activate env
cd throng5 && source .venv/bin/activate

# Train
python -u brain/games/lolo/run_tiered_gan.py

# Export
python cloud/export_weights.py

# Download (from local)
scp -P PORT root@HOST:throng5/outputs/weights/* .
```
