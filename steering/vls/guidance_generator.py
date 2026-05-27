"""VLM-based guidance function generator for VLS steering.

Queries GPT-4o (or any OpenAI-compatible vision model) with the VLS prompt
templates to generate per-stage Python reward functions.  Caches results to
disk so the LLM is only called once per unique (task, scene) combination.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from steering.vls.guidance_utils import load_functions_from_string
from steering.vls.stage_manager import VLSStageSpec

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"


def _encode_image(img_bgr: np.ndarray) -> str:
    """JPEG-encode a BGR image to a base-64 string."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _load_template() -> str:
    base_path = _TEMPLATE_DIR / "guidance_template.txt"
    if not base_path.exists():
        raise FileNotFoundError(f"Guidance template not found: {base_path}")
    base = base_path.read_text()

    # Fill the env-specific slot — empty string if no Isaac template exists
    isaac_path = _TEMPLATE_DIR / "guidance_template_isaac.txt"
    env_notes = isaac_path.read_text() if isaac_path.exists() else ""
    return base.replace("{task_specific_patterns}", env_notes)


def _cache_key(task_name: str, image_rgb: np.ndarray) -> str:
    img_hash = hashlib.md5(image_rgb.tobytes()).hexdigest()[:10]
    return f"{task_name}_{img_hash}"


def _parse_guidance_functions(raw_output: str) -> tuple[list[str], int, list[str]]:
    """Extract per-stage function source strings and metadata from LLM output.

    Returns (function_sources, num_stages, stage_names).
    Each element of function_sources is the complete source text of one
    stageN_guidance function.
    """
    # Extract num_stages
    num_stages = 1
    for line in raw_output.split("\n"):
        m = re.match(r"\s*num_stages\s*=\s*(\d+)", line)
        if m:
            num_stages = int(m.group(1))
            break

    # Split into per-function blocks by 'def stageN_guidance'
    blocks: dict[int, list[str]] = {}
    current_idx: Optional[int] = None
    current_lines: list[str] = []

    for line in raw_output.split("\n"):
        fn_match = re.match(r"def (stage(\d+)_guidance)\(", line)
        if fn_match:
            if current_idx is not None:
                blocks[current_idx] = current_lines
            current_idx = int(fn_match.group(2))
            current_lines = [line]
        elif current_idx is not None:
            current_lines.append(line)
            if line.startswith("    return "):
                blocks[current_idx] = current_lines
                current_idx = None
                current_lines = []

    if current_idx is not None and current_lines:
        blocks[current_idx] = current_lines

    fn_sources = ["\n".join(blocks[i]) for i in sorted(blocks)]

    # Stage names from comments
    stage_names: list[str] = []
    in_breakdown = False
    for line in raw_output.split("\n"):
        if re.search(r"#\s*Task breakdown:", line, re.IGNORECASE):
            in_breakdown = True
            continue
        if in_breakdown:
            m = re.match(r"#\s*Stage\s*\d+:\s*(.+)", line)
            if m:
                stage_names.append(m.group(1).strip())
            elif line.strip() and not line.strip().startswith("#"):
                break

    while len(stage_names) < len(fn_sources):
        stage_names.append(f"Stage {len(stage_names) + 1}")

    return fn_sources, num_stages, stage_names


def _infer_target_keypoint(
    fn_source: str, keypoint_positions: np.ndarray
) -> Optional[np.ndarray]:
    """Heuristic: find the first keypoint index referenced in the function source."""
    m = re.search(r"torch\.tensor\(\[(\d+)\]", fn_source)
    if m:
        idx = int(m.group(1))
        if idx < len(keypoint_positions):
            return keypoint_positions[idx].copy()
    # Fallback: try plain integer literals used for keypoint indexing
    for m in re.finditer(r"keypoints\[(\d+)\]", fn_source):
        idx = int(m.group(1))
        if idx < len(keypoint_positions):
            return keypoint_positions[idx].copy()
    return None


def _infer_primitive(description: str) -> str:
    desc_lower = description.lower()
    if any(w in desc_lower for w in ("grasp", "pick", "grip", "grab")):
        return "grasp"
    if any(w in desc_lower for w in ("place", "put", "drop", "release")):
        return "place"
    return "move"


class VLSGuidanceGenerator:
    """Generate per-stage reward functions for a manipulation task.

    Args:
        llm_model: OpenAI model name (default: ``gpt-4o``).
        temperature: Sampling temperature.
        max_tokens: Max completion tokens.
        cache_dir: Directory for caching LLM outputs.
        api_key: OpenAI API key (falls back to ``OPENAI_API_KEY`` env var).
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 3000,
        cache_dir: str = "outputs/vls_cache",
        api_key: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=key)
        self._model = llm_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cache_dir = Path(cache_dir)
        self._prompt_template = _load_template()

    def generate(
        self,
        instruction: str,
        image_rgb: np.ndarray,
        keypoint_names: list[str],
        keypoint_positions: np.ndarray,
        task_name: str = "task",
    ) -> list[VLSStageSpec]:
        """Generate guidance stages for one episode.

        Args:
            instruction: Language task description.
            image_rgb: (H, W, 3) uint8 RGB image with keypoints overlaid.
            keypoint_names: Human-readable names for each keypoint.
            keypoint_positions: (K, 3) world-frame keypoint positions.
            task_name: Used as the cache key prefix.

        Returns:
            List of VLSStageSpec, one per guidance stage.
        """
        cache_key = _cache_key(task_name, image_rgb)
        cache_path = self._cache_dir / f"{cache_key}_output.txt"

        if cache_path.exists():
            logger.info(f"VLS: loading cached guidance from {cache_path}")
            raw_output = cache_path.read_text()
        else:
            raw_output = self._query_llm(
                instruction, image_rgb, keypoint_names, keypoint_positions
            )
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(raw_output)
            logger.info(f"VLS: cached guidance to {cache_path}")

        return self._build_stages(raw_output, keypoint_positions)

    def _query_llm(
        self,
        instruction: str,
        image_rgb: np.ndarray,
        keypoint_names: list[str],
        keypoint_positions: np.ndarray,
    ) -> str:
        kp_map = {str(i): name for i, name in enumerate(keypoint_names)}
        kp_positions_str = {
            str(i): keypoint_positions[i].tolist()
            for i in range(len(keypoint_positions))
        }

        prompt = self._prompt_template
        prompt = prompt.replace("{instruction}", instruction)
        prompt = prompt.replace("{key_points_objects_map}", str(kp_map))
        prompt = prompt.replace("{init_keypoint_positions}", str(kp_positions_str))
        prompt = prompt.replace("{num_keypoints}", str(len(keypoint_names)))

        img_b64 = _encode_image(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ]

        logger.info(f"VLS: querying {self._model} for '{instruction}'...")
        t0 = time.time()
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
            stream=True,
        )
        output = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                output += chunk.choices[0].delta.content
        logger.info(f"VLS: LLM response received in {time.time() - t0:.1f}s")
        return output

    def _build_stages(
        self, raw_output: str, keypoint_positions: np.ndarray
    ) -> list[VLSStageSpec]:
        fn_sources, num_stages, stage_names = _parse_guidance_functions(raw_output)

        if not fn_sources:
            logger.warning("VLS: no guidance functions parsed from LLM output")
            return []

        stages: list[VLSStageSpec] = []
        for i, (src, name) in enumerate(zip(fn_sources, stage_names)):
            try:
                fns = load_functions_from_string(src, validate=True)
                if not fns:
                    logger.warning(f"VLS: stage {i} function parse failed, skipping")
                    continue
                fn = fns[0]
            except Exception as e:
                logger.error(f"VLS: stage {i} function load failed: {e}")
                continue

            target = _infer_target_keypoint(src, keypoint_positions)
            primitive = _infer_primitive(name)
            stages.append(
                VLSStageSpec(
                    guidance_fn=fn,
                    target_world=target,
                    primitive=primitive,
                    description=name,
                )
            )
            logger.info(
                f"VLS: stage {i} [{primitive}] '{name}' "
                f"target={target.tolist() if target is not None else None}"
            )

        return stages
