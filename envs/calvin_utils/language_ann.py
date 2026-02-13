"""Language annotation loading for CALVIN dataset.

Loads language annotations from auto_lang_ann.npy file and creates
task name to instruction mappings.
"""

import os
import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


def load_language_annotations(ann_path: str) -> Dict[str, str]:
    """
    Load CALVIN language annotations and create task -> instruction mapping.

    The CALVIN dataset provides language annotations in the format:
    {
        'language': {
            'ann': [list of instruction strings],
            'task': [list of task IDs],
            'emb': [precomputed embeddings]
        },
        'info': {
            'indx': [start/end indices for sequences]
        }
    }

    Args:
        ann_path: Path to auto_lang_ann.npy file

    Returns:
        Dictionary mapping task names/IDs to language instructions
        Empty dict if file not found
    """
    if ann_path is None or not os.path.exists(ann_path):
        logger.warning(f"Language annotation file not found: {ann_path}")
        logger.info("Falling back to task names as instructions")
        return {}

    try:
        ann_data = np.load(ann_path, allow_pickle=True).item()
        logger.info(f"Loaded language annotations from: {ann_path}")

        # Extract language and task information
        if 'language' not in ann_data:
            logger.error("Invalid annotation file format: missing 'language' key")
            return {}

        language_data = ann_data['language']
        if 'ann' not in language_data or 'task' not in language_data:
            logger.error("Invalid annotation file format: missing 'ann' or 'task' keys")
            return {}

        annotations = language_data['ann']
        task_ids = language_data['task']

        # Create task -> instruction mapping
        # Use first annotation for each unique task
        task_to_instruction = {}
        for task_id, instruction in zip(task_ids, annotations):
            task_key = str(task_id)  # Ensure string key
            if task_key not in task_to_instruction:
                task_to_instruction[task_key] = instruction
                logger.debug(f"Task '{task_key}': {instruction}")

        logger.info(f"Loaded {len(task_to_instruction)} unique task annotations")
        return task_to_instruction

    except Exception as e:
        logger.error(f"Error loading language annotations: {e}")
        return {}


def get_instruction_for_task(task_name: str, task_instructions: Dict[str, str]) -> str:
    """
    Get language instruction for a specific task.

    Args:
        task_name: Task name/ID
        task_instructions: Dictionary of task -> instruction mappings

    Returns:
        Language instruction string, or task_name if not found
    """
    # Try direct lookup
    if task_name in task_instructions:
        return task_instructions[task_name]

    # Try case-insensitive lookup
    task_name_lower = task_name.lower()
    for key, instruction in task_instructions.items():
        if key.lower() == task_name_lower:
            return instruction

    # Fallback to task name
    logger.warning(f"No annotation found for task '{task_name}', using task name as instruction")
    return task_name
