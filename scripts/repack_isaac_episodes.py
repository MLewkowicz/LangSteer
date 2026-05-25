"""Repack Isaac Sim HDF5 episodes for fast random-access reads.

The recorder writes episodes with gzip-1 compression and h5py's default
chunking heuristic.  At training time, IsaacDataset reads only chunk_size
frames per __getitem__ via integer indexing, which forces h5py to decompress
whole chunks just to extract a few frames.  On a shared filesystem with 32
worker processes (4 GPU × 8 workers) all hammering the same files at once,
this becomes the dominant bottleneck — GPU power readings of ~25% are the
tell.

This script rewrites each episode_*.h5 in place with:
  * compression=None  (no gzip decode on the read path)
  * chunks=(1, H, W, 3)  for RGB/PCD arrays  (one chunk == one frame, so
                          every __getitem__ reads exactly the bytes it needs)
  * everything else unchunked

Trade-off: files grow ~3× (gzip-1 on uint8 RGB and float32 PCD compresses
moderately).  At 27 GB → ~80 GB, well within typical /data/scratch quotas.

Usage::

    uv run python scripts/repack_isaac_episodes.py \\
        --data-dir /data/scratch/mlewkowicz/isaac_sim_demos
    # add --dry-run first if you want to see what it'd do
"""

import argparse
import shutil
import time
from pathlib import Path

import h5py
import numpy as np


def repack_one(src: Path, dst_tmp: Path) -> tuple[int, int]:
    """Repack one episode to dst_tmp. Returns (src_bytes, dst_bytes)."""
    with h5py.File(src, "r") as fin, h5py.File(dst_tmp, "w") as fout:
        for name in fin.keys():
            arr = fin[name][:]
            # Per-frame chunking for the (T, 200, 200, 3) RGB/PCD arrays.
            # Anything else gets unchunked storage.
            if arr.ndim == 4 and arr.shape[1:3] == (200, 200):
                chunks = (1,) + arr.shape[1:]
            else:
                chunks = None
            fout.create_dataset(
                name, data=arr, chunks=chunks, compression=None,
            )
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
    return src.stat().st_size, dst_tmp.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-dir", required=True, type=Path,
        help="Top-level dir containing episode_*.h5 (recursively).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print plan; do not write or replace anything.",
    )
    args = ap.parse_args()

    if not args.data_dir.is_dir():
        ap.error(f"{args.data_dir} is not a directory")

    # `episode_*.h5` matches files written by IsaacSimRecorder; rglob covers
    # both training/ and validation/ subdirs.
    episodes = sorted(p for p in args.data_dir.rglob("episode_*.h5") if p.is_file())
    if not episodes:
        print(f"No episode_*.h5 files found under {args.data_dir}")
        return

    print(f"Found {len(episodes)} episodes under {args.data_dir}")
    if args.dry_run:
        for p in episodes[:5]:
            print(f"  would repack {p}")
        if len(episodes) > 5:
            print(f"  … and {len(episodes) - 5} more")
        return

    t0 = time.monotonic()
    total_old = total_new = 0
    failed: list[Path] = []
    for i, src in enumerate(episodes, 1):
        tmp = src.with_suffix(".repacked.h5")
        try:
            old_b, new_b = repack_one(src, tmp)
        except Exception as e:
            print(f"  [{i}/{len(episodes)}] FAIL {src.name}: {e}")
            if tmp.exists():
                tmp.unlink()
            failed.append(src)
            continue
        # Atomic replace: the rename guarantees the dataset is never half-
        # written under the original name.
        tmp.replace(src)
        total_old += old_b
        total_new += new_b
        if i == 1 or i % 10 == 0 or i == len(episodes):
            rate = i / max(time.monotonic() - t0, 1e-6)
            print(
                f"  [{i}/{len(episodes)}] {src.name}: "
                f"{old_b // (1024 * 1024)}MB → {new_b // (1024 * 1024)}MB  "
                f"({rate:.2f} files/s)"
            )

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(
        f"Total: {total_old / 1024**3:.1f} GB → {total_new / 1024**3:.1f} GB "
        f"({100 * total_new / max(total_old, 1):.0f}%)"
    )
    if failed:
        print(f"{len(failed)} episodes failed to repack:")
        for p in failed:
            print(f"  {p}")


if __name__ == "__main__":
    main()
