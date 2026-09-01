"""Filter a packaged DiffuserActor dataset to episodes whose name starts with a prefix.

The dataset loader reads `episode[6][0]` from each .dat to index into the
`primitive_ids` / `object_ids` arrays built from the lang annotation npy
(`dataset.py:487-505`), so a faithful filter must:

  1. Drop rows from `primitive_object_lang_ann.npy` whose `info.episodes[i]`
     does not match the prefix.
  2. Renumber the kept .dat files to ann_0.dat..ann_{N-1}.dat.
  3. Rewrite each kept .dat's `episode[6]` (the per-frame ann-id pointer) to
     its new row index so it lines up with the filtered npy.

Outputs a new packaged tree mirroring the source layout. Both training/ and
validation/ splits are filtered (and either may end up empty).

Usage:
    uv run python scripts/filter_dat_by_episode_prefix.py \\
        --src /home/clear/Documents/michal/realworld_3da_v2 \\
        --dst /home/clear/Documents/michal/realworld_3da_v2_cabinet \\
        --prefix cabinet_
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import blosc
import numpy as np


SPLITS = ("training", "validation")
DAT_SUBDIR = "D+0"
ANN_FILE = "lang_annotations/primitive_object_lang_ann.npy"


def _filter_one_split(src_split: Path, dst_split: Path, prefix: str | None,
                      keep_range: tuple[int, int] | None) -> int:
    ann_path = src_split / ANN_FILE
    if not ann_path.is_file():
        print(f"  [{src_split.name}] no annotation file, skip")
        return 0
    ann = np.load(ann_path, allow_pickle=True).item()
    info = ann["info"]
    lang = ann["language"]
    n_old = len(info["primitive"])

    if keep_range is not None:
        lo, hi = keep_range
        keep_old = [i for i in range(max(0, lo), min(n_old, hi))]
        criterion = f"index range [{lo}:{hi})"
    else:
        episodes = list(info.get("episodes", []))
        if len(episodes) != n_old:
            raise RuntimeError(
                f"info.episodes missing/short in {ann_path} "
                f"({len(episodes)} != {n_old}); pass --keep_train_range / "
                f"--keep_val_range to filter by index instead.")
        keep_old = [i for i, name in enumerate(episodes) if str(name).startswith(prefix)]
        criterion = f"prefix '{prefix}'"
    n_new = len(keep_old)
    print(f"  [{src_split.name}] {n_new}/{n_old} segments match {criterion}")
    if n_new == 0:
        return 0

    # --- New annotation file with rows sliced to kept indices --------------
    def _slice(seq):
        """Slice if length matches n_old; otherwise pass through (handles
        legacy npys where some info fields are empty, e.g. `episodes`)."""
        if isinstance(seq, np.ndarray):
            return seq[keep_old] if seq.shape and seq.shape[0] == n_old else seq
        if isinstance(seq, list):
            return [seq[i] for i in keep_old] if len(seq) == n_old else list(seq)
        return seq
    new_info = {k: _slice(v) for k, v in info.items()}
    new_lang = {k: _slice(v) for k, v in lang.items()}
    new_payload = {"language": new_lang, "info": new_info}

    out_ann = dst_split / ANN_FILE
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_ann, new_payload, allow_pickle=True)

    # --- Copy each kept .dat, renumbered + ann-id rewritten ----------------
    out_dat_dir = dst_split / DAT_SUBDIR
    out_dat_dir.mkdir(parents=True, exist_ok=True)
    for new_i, old_i in enumerate(keep_old):
        src_dat = src_split / DAT_SUBDIR / f"ann_{old_i}.dat"
        if not src_dat.is_file():
            print(f"    MISSING {src_dat.name} (skipping)")
            continue
        episode = pickle.loads(blosc.decompress(src_dat.read_bytes()))
        # episode[6] is a list of identical ann-ids, one per kept frame.
        if isinstance(episode, list):
            episode = list(episode)
            episode[6] = [new_i] * len(episode[6])
        out_dat = out_dat_dir / f"ann_{new_i}.dat"
        out_dat.write_bytes(blosc.compress(pickle.dumps(episode), typesize=8))
    print(f"  [{src_split.name}] wrote {n_new} .dat files + npy → {dst_split}")
    return n_new


def _parse_range(s: str | None) -> tuple[int, int] | None:
    if s is None:
        return None
    lo_s, hi_s = s.split(":")
    return int(lo_s or 0), int(hi_s) if hi_s else 10**9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="Existing packaged dataset (with training/ and validation/).")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Destination dir (will be created or overwritten).")
    ap.add_argument("--prefix", default=None,
                    help="Keep segments whose `info.episodes[i]` starts with this. "
                         "Requires info.episodes to be populated (post-patch converter).")
    ap.add_argument("--keep_train_range", default=None,
                    help="Fallback for legacy npys without info.episodes: half-open "
                         "index range '[lo:hi)' on the training split, e.g. '0:76'.")
    ap.add_argument("--keep_val_range", default=None,
                    help="Same idea for the validation split. Pass an empty range "
                         "(e.g. '0:0') to drop the split entirely.")
    args = ap.parse_args()

    if args.prefix is None and args.keep_train_range is None and args.keep_val_range is None:
        ap.error("pass either --prefix or --keep_{train,val}_range.")

    ranges = {
        "training": _parse_range(args.keep_train_range),
        "validation": _parse_range(args.keep_val_range),
    }
    total = 0
    for split in SPLITS:
        total += _filter_one_split(args.src / split, args.dst / split,
                                   args.prefix, ranges[split])
    print(f"\nDone. {total} total segments under {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
