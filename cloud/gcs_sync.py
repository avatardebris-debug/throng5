"""
gcs_sync.py — Upload/download weights and data to/from Google Cloud Storage.

Transfers happen ONLY before/after training — zero overhead during training.

Usage:
  # Download weights before training:
  python cloud/gcs_sync.py pull

  # Upload weights after training:
  python cloud/gcs_sync.py push

  # Push only weights (skip replay DB):
  python cloud/gcs_sync.py push --weights-only

  # Pull only replay DB:
  python cloud/gcs_sync.py pull --replay-only

Setup:
  1. Install: pip install google-cloud-storage
  2. Auth:   gcloud auth application-default login
     OR set: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
  3. Set bucket name below or via env: GCS_BUCKET=your-bucket-name
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ── Configuration ─────────────────────────────────────────────────────

# Set your GCS bucket name here or via env var GCS_BUCKET
DEFAULT_BUCKET = os.environ.get("GCS_BUCKET", "throng5-weightsb")

# What to sync
WEIGHT_FILES = [
    "brain/games/lolo/dqn_weights.pt",
    "brain/games/lolo/sarsa_qtable.npy",
    "brain/games/lolo/gan_checkpoint.json",
]

REPLAY_FILES = [
    "throng3 - Copy/experiments/replay_db.sqlite",
    "brain/games/lolo/training_stats.json",
]

GCS_PREFIX = "lolo/"  # Folder inside bucket


def get_storage_client():
    """Get GCS client, with helpful error if not installed."""
    try:
        from google.cloud import storage
        return storage.Client()
    except ImportError:
        print("ERROR: google-cloud-storage not installed.")
        print("  pip install google-cloud-storage")
        print("  gcloud auth application-default login")
        sys.exit(1)


def upload_file(client, bucket_name: str, local_path: str, gcs_path: str):
    """Upload a single file to GCS."""
    if not os.path.exists(local_path):
        print(f"  ⚠ Skip (not found): {local_path}")
        return False

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  ↑ Uploading {local_path} ({size_mb:.1f} MB)...", end="", flush=True)

    t0 = time.time()
    blob.upload_from_filename(local_path)
    elapsed = time.time() - t0

    print(f" done ({elapsed:.1f}s)")
    return True


def download_file(client, bucket_name: str, gcs_path: str, local_path: str):
    """Download a single file from GCS."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        print(f"  ⚠ Skip (not in GCS): {gcs_path}")
        return False

    # Create parent dirs
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    print(f"  ↓ Downloading {gcs_path}...", end="", flush=True)

    t0 = time.time()
    blob.download_to_filename(local_path)
    elapsed = time.time() - t0

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f" done ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return True


def push(bucket_name: str, weights_only: bool = False):
    """Upload weights (and optionally replay DB) to GCS."""
    client = get_storage_client()

    print(f"\n  Pushing to gs://{bucket_name}/{GCS_PREFIX}")
    print(f"  {'─' * 50}")

    uploaded = 0
    for f in WEIGHT_FILES:
        gcs_path = GCS_PREFIX + os.path.basename(f)
        if upload_file(client, bucket_name, f, gcs_path):
            uploaded += 1

    if not weights_only:
        for f in REPLAY_FILES:
            gcs_path = GCS_PREFIX + "replay/" + os.path.basename(f)
            if upload_file(client, bucket_name, f, gcs_path):
                uploaded += 1

    print(f"\n  ✅ Pushed {uploaded} files to gs://{bucket_name}/{GCS_PREFIX}")


def pull(bucket_name: str, weights_only: bool = False, replay_only: bool = False):
    """Download weights (and optionally replay DB) from GCS."""
    client = get_storage_client()

    print(f"\n  Pulling from gs://{bucket_name}/{GCS_PREFIX}")
    print(f"  {'─' * 50}")

    downloaded = 0

    if not replay_only:
        for f in WEIGHT_FILES:
            gcs_path = GCS_PREFIX + os.path.basename(f)
            if download_file(client, bucket_name, gcs_path, f):
                downloaded += 1

    if not weights_only:
        for f in REPLAY_FILES:
            gcs_path = GCS_PREFIX + "replay/" + os.path.basename(f)
            if download_file(client, bucket_name, gcs_path, f):
                downloaded += 1

    print(f"\n  ✅ Pulled {downloaded} files from gs://{bucket_name}/{GCS_PREFIX}")


def list_remote(bucket_name: str):
    """List all files in the GCS bucket prefix."""
    client = get_storage_client()
    bucket = client.bucket(bucket_name)

    print(f"\n  Files in gs://{bucket_name}/{GCS_PREFIX}")
    print(f"  {'─' * 50}")

    blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
    if not blobs:
        print("  (empty)")
        return

    for blob in blobs:
        size_mb = blob.size / (1024 * 1024) if blob.size else 0
        print(f"  {blob.name:50s} {size_mb:8.1f} MB  {blob.updated}")


def main():
    parser = argparse.ArgumentParser(description="Sync weights with Google Cloud Storage")
    parser.add_argument("action", choices=["push", "pull", "list"],
                        help="push=upload, pull=download, list=show remote files")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET,
                        help=f"GCS bucket name (default: {DEFAULT_BUCKET})")
    parser.add_argument("--weights-only", action="store_true",
                        help="Only sync weight files (skip replay DB)")
    parser.add_argument("--replay-only", action="store_true",
                        help="Only sync replay DB (skip weights)")
    args = parser.parse_args()

    if args.action == "push":
        push(args.bucket, weights_only=args.weights_only)
    elif args.action == "pull":
        pull(args.bucket, weights_only=args.weights_only,
             replay_only=args.replay_only)
    elif args.action == "list":
        list_remote(args.bucket)


if __name__ == "__main__":
    main()
