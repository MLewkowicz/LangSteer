"""
Convert a CALVIN dataset split to LeRobot format for pi-0.5 fine-tuning.

CALVIN stores one frame per .npz file. Episodes are defined by language annotation
ranges in lang_annotations/auto_lang_ann.npy. This script walks every annotation,
loads the corresponding frames, and writes them into a LeRobotDataset.

Usage:
    uv run python training/common/convert_calvin_to_lerobot.py \
        --data_dir /path/to/task_ABC_D \
        --repo_id your_hf_username/calvin_abc_d

    # Convert validation split too (run separately):
    uv run python training/common/convert_calvin_to_lerobot.py \
        --data_dir /path/to/task_ABC_D \
        --repo_id your_hf_username/calvin_abc_d \
        --split validation

    # Push to Hugging Face Hub:
    uv run python training/common/convert_calvin_to_lerobot.py \
        --data_dir /path/to/task_ABC_D \
        --repo_id your_hf_username/calvin_abc_d \
        --push_to_hub

repo_id format: "your_hf_username/dataset_name"
  - Used as the local directory name under $HF_LEROBOT_HOME
    (default: ~/.cache/huggingface/lerobot/)
  - Also used as the Hugging Face Hub repository name if --push_to_hub is set

Prerequisites:
    uv pip install lerobot tqdm
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    # lerobot < 0.4
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ImportError:
    # lerobot >= 0.4
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    except ImportError:
        try:
            from lerobot.constants import HF_LEROBOT_HOME
        except ImportError:
            HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


FEATURES = {
    "image": {
        "dtype": "image",
        "shape": (200, 200, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (84, 84, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (15,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"],
    },
}


@dataclass
class Args:
    data_dir: str
    """Path to CALVIN dataset root (e.g. ~/calvin/dataset/task_ABC_D)."""

    repo_id: str
    """HuggingFace repo id in 'username/dataset_name' format. Controls both
    the local output path ($HF_LEROBOT_HOME/repo_id) and the Hub destination."""

    split: str = "training"
    """Dataset split to convert. Run the script twice to convert both splits."""

    push_to_hub: bool = False
    """Push the converted dataset to the Hugging Face Hub."""

    image_writer_threads: int = 0
    image_writer_processes: int = 0


def _decode(s) -> str:
    return s.decode() if isinstance(s, bytes) else s


def main(args: Args) -> None:
    data_dir = Path(args.data_dir).expanduser()
    split_dir = data_dir / args.split

    # Remove any existing local copy
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    ann_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    annotations = np.load(ann_path, allow_pickle=True).item()
    indx = annotations["info"]["indx"]
    ann_texts = annotations["language"]["ann"]

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="unknown",
        fps=30,
        features=FEATURES,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    for i, (start_id, end_id) in tqdm(
        enumerate(indx), total=len(indx), desc=f"Converting {args.split}"
    ):
        instruction = _decode(ann_texts[i])
        frames_added = 0

        for frame_id in range(start_id, end_id + 1):
            episode_path = split_dir / f"episode_{frame_id:07d}.npz"
            if not episode_path.exists():
                continue

            npz = np.load(episode_path)
            dataset.add_frame(
                {
                    "image": npz["rgb_static"],
                    "wrist_image": npz["rgb_gripper"],
                    "state": npz["robot_obs"].astype(np.float32),
                    "action": npz["rel_actions"].astype(np.float32),
                    "task": instruction,
                }
            )
            frames_added += 1

        if frames_added > 0:
            dataset.save_episode()

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["calvin", "manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
