"""Precompute CLIP instruction embeddings for CALVIN dataset.

Adapted from 3d_diffuser_actor/data_preprocessing/preprocess_calvin_instructions.py.
Replaces tap.Tap with a plain function for integration with the convert script.

Usage (standalone):
    python -m training.policies.diffuser_actor.preprocessing.preprocess_instructions \
        --annotation_path /path/to/task_D_D/training/lang_annotations/auto_lang_ann.npy \
        --output /path/to/output/training.pkl

Usage (from convert_calvin_for_diffuser_actor.py):
    Called via main(tasks_path, output_path) which processes both splits.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm.auto import tqdm


def load_model(encoder="clip", device="cuda"):
    """Load pretrained text encoder."""
    if encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    elif encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    return model.to(device)


def load_tokenizer(encoder="clip", model_max_length=53):
    """Load pretrained tokenizer."""
    if encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    elif encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    tokenizer.model_max_length = model_max_length
    return tokenizer


def precompute_embeddings(annotation_path, output_path, encoder="clip",
                          model_max_length=53, device="cuda"):
    """Precompute text embeddings for a single annotation file.

    Args:
        annotation_path: Path to auto_lang_ann.npy
        output_path: Path for output .pkl file
        encoder: "clip" or "bert"
        model_max_length: Max token length
        device: torch device
    """
    annotations = np.load(str(annotation_path), allow_pickle=True).item()
    instructions_string = [s + '.' for s in annotations['language']['ann']]

    tokenizer = load_tokenizer(encoder, model_max_length)
    model = load_model(encoder, device)

    instructions = {
        'embeddings': [],
        'text': []
    }

    for instr in tqdm(instructions_string, desc="Encoding instructions"):
        tokens = tokenizer(instr, padding="max_length")["input_ids"]
        tokens = torch.tensor(tokens).to(device).view(1, -1)
        with torch.no_grad():
            pred = model(tokens).last_hidden_state
        instructions['embeddings'].append(pred.cpu())
        instructions['text'].append(instr)

    os.makedirs(str(Path(output_path).parent), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(instructions, f)
    print(f"Saved {len(instructions_string)} embeddings to {output_path}")


def main(tasks_path, output_path):
    """Process both training and validation splits."""
    tasks_path = Path(tasks_path)
    output_path = Path(output_path)

    for split in ["training", "validation"]:
        ann_path = tasks_path / split / "lang_annotations" / "auto_lang_ann.npy"
        if not ann_path.exists():
            print(f"Skipping {split}: {ann_path} not found")
            continue
        out_path = output_path / f"{split}.pkl"
        print(f"Processing {split} split...")
        precompute_embeddings(ann_path, out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute CLIP instruction embeddings for CALVIN"
    )
    parser.add_argument(
        "--annotation_path", type=str, required=True,
        help="Path to auto_lang_ann.npy"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output .pkl path"
    )
    parser.add_argument("--encoder", type=str, default="clip")
    parser.add_argument("--model_max_length", type=int, default=53)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    precompute_embeddings(
        args.annotation_path, args.output,
        encoder=args.encoder,
        model_max_length=args.model_max_length,
        device=args.device,
    )
