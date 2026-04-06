#!/usr/bin/env python3
"""Convert CALVIN dataset for 3D Diffuser Actor training.

This script packages CALVIN episodes into .dat files with blosc compression,
then optionally precomputes CLIP instruction embeddings.

Usage:
    # Package all episodes + instructions
    python scripts/convert_calvin_for_diffuser_actor.py \
        --tasks_path /path/to/task_D_D \
        --output_path /path/to/output

    # Package only specific tasks (no language embeddings)
    python scripts/convert_calvin_for_diffuser_actor.py \
        --tasks_path /path/to/task_D_D \
        --output_path /path/to/output \
        --tasks open_drawer move_slider_left \
        --skip_instructions

    # Precompute CLIP instructions only
    python scripts/convert_calvin_for_diffuser_actor.py \
        --precompute_instructions \
        --tasks_path /path/to/task_D_D \
        --output_path /path/to/output
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Convert CALVIN dataset for 3D Diffuser Actor"
    )
    parser.add_argument(
        "--precompute_instructions", action="store_true",
        help="Only precompute CLIP instruction embeddings"
    )
    parser.add_argument(
        "--tasks_path", type=str, required=True,
        help="Path to CALVIN task_D_D directory"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output directory for packaged data"
    )
    parser.add_argument(
        "--num_workers", type=int, default=10,
        help="Number of parallel workers for episode packaging"
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Filter to specific CALVIN task names (e.g., open_drawer move_slider_left)"
    )
    parser.add_argument(
        "--skip_instructions", action="store_true",
        help="Skip CLIP instruction embedding precomputation (for non-language training)"
    )

    args = parser.parse_args()

    if args.precompute_instructions:
        from training.policies.diffuser_actor.preprocessing.preprocess_instructions import (
            main as preprocess_main,
        )
        preprocess_main(args.tasks_path, args.output_path)
    else:
        from training.policies.diffuser_actor.preprocessing.package_calvin import (
            main as package_main,
        )
        for split in ["training", "validation"]:
            package_main(
                split=split,
                root_dir=args.tasks_path,
                save_path=args.output_path,
                tasks=args.tasks,
            )

        if not args.skip_instructions:
            from training.policies.diffuser_actor.preprocessing.preprocess_instructions import (
                main as preprocess_main,
            )
            preprocess_main(args.tasks_path, args.output_path)


if __name__ == "__main__":
    main()
