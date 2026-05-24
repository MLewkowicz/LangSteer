"""Language Model Program (LMP) engine for VoxPoser value map generation.

Modernized from VoxPoser/src/LMP.py with provider-agnostic LLM backends
(Anthropic Claude and OpenAI GPT). Uses in-context learning prompts to
generate Python code for 3D value map construction.

`compose_with_repair` wraps the composer LMP with a vocab-adherence linter:
it validates emitted stages against `PRIMITIVE_VOCAB` / `OBJECT_VOCAB` (single
source of truth in `steering.stage_spec`) and re-prompts the LLM with explicit
fix hints when stages would otherwise be dropped by `parse_composer_stages`.
On exhaustion (`max_repromptings` consecutive invalid emissions) it raises
`VocabValidationError` — no silent fallback.
"""

import logging
import time
from difflib import get_close_matches
from typing import Any, Dict, Optional

import numpy as np

# NOTE: `steering.stage_spec`'s vocab dicts are imported lazily inside the
# linter helpers below to avoid a circular import. `steering/__init__.py`
# eagerly loads `steering.voxposer_steering` → `steering.stage_manager` →
# `voxposer.lmp` (this module), so a top-level `from steering...` here would
# pick up a half-initialized `steering` package.

from voxposer.calvin_interface import CalvinLMPInterface
from voxposer.llm_cache import DiskCache
from voxposer.utils import (
    DynamicObservation,
    IterableDynamicObservation,
    Observation,
    load_prompt,
)

logger = logging.getLogger(__name__)


class VocabValidationError(ValueError):
    """Raised when the composer keeps emitting invalid vocab after re-prompts.

    Carries the final violations list + raw result so callers can log the
    failure mode without re-parsing. Subclass of ValueError so existing
    `except ValueError` clauses still catch it; StageManager catches Exception
    broadly but narrows to let this one propagate (no silent fallback —
    failures must reach the runner per the user's "no hacking" rule).
    """


# Composer-emitted aliases for vocabulary terms that the LLM consistently
# confuses with valid ones. Audit (2026-05-18) caught 'door_handle' on
# move_slider_* — gpt-4o emits it because the canonical CALVIN instruction is
# "slide the door to the left" and "door handle" is the natural phrasing.
# Extend as new confusion modes surface in subsequent audits.
HANDLE_ALIASES: Dict[str, str] = {
    "door_handle": "slider_handle",
}

# System prompt for the LLM (shared across providers)
SYSTEM_PROMPT = (
    "You are a helpful assistant that writes Python code to control a robot arm "
    "in a tabletop manipulation environment (CALVIN benchmark). Complete the code "
    "when given a new query. Follow the patterns in the context code. Be thorough "
    "and thoughtful. Do not include import statements. Do not repeat the query. "
    "Do not provide text explanations (comments in code are okay). "
    "Note: x is left(-) to right(+), y is front(-) to back(+), z is bottom to top."
)


class LLMBackend:
    """Provider-agnostic LLM API wrapper."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        cache: DiskCache,
    ):
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cache = cache
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self._provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        elif self._provider == "openai":
            import openai

            self._client = openai.OpenAI()
        else:
            raise ValueError(f"Unknown LLM provider: {self._provider}")
        return self._client

    def generate(self, prompt: str, stop: list, image_bytes: Optional[bytes] = None) -> str:
        """Generate code completion from the LLM, with caching.

        Task 7 Phase 2 extension: when `image_bytes` is provided, the call routes
        to the OpenAI vision content shape (list of `{type:text/image_url}`
        items). The cache key gains an `image_sha256` field so vision calls
        live in a separate namespace from text-only calls. Existing text-only
        callers continue to hit the same cache entries as today.
        """
        cache_key = {
            "provider": self._provider,
            "model": self._model,
            "prompt": prompt,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if image_bytes is not None:
            import hashlib
            cache_key["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()

        if cache_key in self._cache:
            logger.debug("Using cached LLM response")
            return self._cache[cache_key]

        client = self._get_client()
        start_time = time.time()

        # Split prompt into context and query for chat-style APIs
        parts = prompt.rsplit("# Query:", 1)
        if len(parts) == 2:
            context = parts[0].strip()
            query = "# Query:" + parts[1]
        else:
            context = prompt
            query = ""

        # Retry with exponential backoff
        for attempt in range(5):
            try:
                if self._provider == "anthropic":
                    if image_bytes is not None:
                        raise NotImplementedError(
                            "Anthropic multimodal deferred to a future task; "
                            "use provider=openai for vision calls."
                        )
                    result = self._call_anthropic(client, context, query, stop)
                else:
                    result = self._call_openai(
                        client, context, query, stop, image_bytes=image_bytes
                    )
                break
            except Exception as e:
                wait = 2**attempt
                logger.warning(
                    f"LLM API error (attempt {attempt + 1}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
        else:
            raise RuntimeError("LLM API failed after 5 attempts")

        elapsed = time.time() - start_time
        logger.info(
            f"LLM API call ({self._provider}/{self._model}) took {elapsed:.2f}s"
        )

        self._cache[cache_key] = result
        return result

    def _call_anthropic(self, client, context: str, query: str, stop: list) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    f"I will give you context code, then a query to complete.\n\n"
                    f"Context:\n```\n{context}\n```\n\n"
                    f"Complete this:\n{query}"
                ),
            },
        ]
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=SYSTEM_PROMPT,
            stop_sequences=stop if stop else None,
            messages=messages,
        )
        text = response.content[0].text.strip()
        # Clean markdown fences if present
        text = text.replace("```python", "").replace("```", "").strip()
        return text

    def _call_openai(
        self, client, context: str, query: str, stop: list,
        image_bytes: Optional[bytes] = None,
    ) -> str:
        user_text = (
            f"I will give you context code, then a query to complete.\n\n"
            f"Context:\n```\n{context}\n```\n\n"
            f"Complete this:\n{query}"
        )
        if image_bytes is not None:
            # OpenAI multimodal content shape (Task 7 Phase 2). The image is
            # passed inline as a base64 data URL; `detail: high` preserves
            # spatial information from the annotated overhead frame.
            import base64
            b64 = base64.b64encode(image_bytes).decode("ascii")
            user_content: Any = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                },
            ]
        else:
            user_content = user_text
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        # Per-model parameter compatibility. gpt-4o accepts the legacy
        # `max_tokens` + `stop=[...]` shape. gpt-5 family (mini/.4-mini/etc.)
        # rejects `max_tokens` (requires `max_completion_tokens`) and rejects
        # `stop` (no server-side termination); for those models we truncate
        # at the stop sequence client-side.
        is_gpt5_family = self._model.startswith("gpt-5")
        kwargs: dict = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": messages,
        }
        if is_gpt5_family:
            kwargs["max_completion_tokens"] = self._max_tokens
        else:
            kwargs["max_tokens"] = self._max_tokens
            kwargs["stop"] = stop if stop else None
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content.strip()
        text = text.replace("```python", "").replace("```", "").strip()
        # Client-side stop-sequence truncation for models that don't support
        # server-side `stop` (gpt-5 family). Mirrors the original behavior:
        # the LLM would have stopped at the first occurrence of any token in
        # `stop`; we slice the text the same way.
        if is_gpt5_family and stop:
            for marker in stop:
                idx = text.find(marker)
                if idx >= 0:
                    text = text[:idx].rstrip()
        return text


class LMP:
    """Language Model Program: generates and executes Python code from LLM.

    Each LMP instance corresponds to one skill (composer, get_affordance_map,
    parse_query_obj, etc.). It loads a prompt template, sends queries to the
    LLM, and safely executes the returned code.
    """

    def __init__(
        self,
        name: str,
        cfg: dict,
        fixed_vars: dict,
        variable_vars: dict,
        backend: LLMBackend,
        env_name: str = "calvin",
    ):
        self._name = name
        self._cfg = cfg
        self._backend = backend
        self._base_prompt = load_prompt(f"{env_name}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg.get("stop", ["# Query:"]))
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ""
        self._context = None
        # Task 7 Phase 2 expansion — runtime-injected scene text (state +
        # VLM grounding). Prepended to the query for downstream LMPs
        # (affordance / avoidance) that don't go through compose_with_repair.
        self._scene_text: Optional[str] = None

    def clear_exec_hist(self):
        self.exec_hist = ""

    def build_prompt(self, query: str):
        """Build the full prompt from template + session history + query."""
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = (
                f"from utils import {', '.join(self._variable_vars.keys())}"
            )
        else:
            variable_vars_imports_str = ""

        prompt = self._base_prompt.replace(
            "{variable_vars_imports}", variable_vars_imports_str
        )

        if self._cfg.get("maintain_session") and self.exec_hist:
            prompt += f"\n{self.exec_hist}"

        prompt += "\n"

        if self._cfg.get("include_context") and self._context is not None:
            prompt += f"\n{self._context}"

        # Task 7 Phase 2 expansion — prepend scene text (state + VLM dict)
        # to the query when set. Affordance / avoidance LMPs read this; the
        # composer receives the same text via `compose_with_repair`'s
        # `scene_context` kwarg, so no double-injection there.
        query_prefix = self._cfg.get("query_prefix", "# Query: ")
        query_suffix = self._cfg.get("query_suffix", ".")
        if self._scene_text:
            user_query = f"{self._scene_text}\n{query_prefix}{query}{query_suffix}"
        else:
            user_query = f"{query_prefix}{query}{query_suffix}"
        prompt += f"\n{user_query}"

        return prompt, user_query

    def __call__(self, query: str, image_bytes: Optional[bytes] = None, **kwargs):
        """Generate code for the query, execute it, and return the result.

        Task 7 Phase 2: `image_bytes` is threaded to the backend for multimodal
        LMPs (currently only `scene_grounding`). Text-only LMPs leave it `None`
        and the backend behaves identically to pre-Task-7.
        """
        prompt, user_query = self.build_prompt(query)

        code_str = self._backend.generate(
            prompt, self._stop_tokens, image_bytes=image_bytes,
        )

        if self._cfg.get("include_context") and self._context is not None:
            to_exec = f"{self._context}\n{code_str}"
            to_log = f"{self._context}\n{user_query}\n{code_str}"
        else:
            to_exec = code_str
            to_log = f"{user_query}\n{to_exec}"

        logger.info(f'[LMP "{self._name}"] generated code:\n{to_log}')

        gvars = {**self._fixed_vars, **self._variable_vars}
        lvars = kwargs

        # Non-composer LMPs return functions for lazy evaluation. Exceptions:
        # composer (returns the stage list directly), planner (returns the
        # task plan), and scene_grounding (Task 7 Phase 2 — returns a dict).
        if self._name not in ["composer", "planner", "scene_grounding"]:
            to_exec = "def ret_val():\n" + to_exec.replace("ret_val = ", "return ")
            to_exec = to_exec.replace("\n", "\n    ")

        exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f"\n{to_log.strip()}"

        if self._cfg.get("maintain_session"):
            self._variable_vars.update(lvars)

        if self._cfg.get("has_return"):
            return_val_name = self._cfg.get("return_val_name", "ret_val")
            if self._name == "parse_query_obj":
                try:
                    return IterableDynamicObservation(lvars[return_val_name])
                except (AssertionError, AssertionError):
                    return DynamicObservation(lvars[return_val_name])
            return lvars[return_val_name]


def exec_safe(
    code_str: str, gvars: Optional[dict] = None, lvars: Optional[dict] = None
):
    """Execute code string in a sandboxed environment.

    Bans import statements and dunder access for safety.
    """
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        if phrase in code_str:
            raise ValueError(
                f"Banned phrase '{phrase}' found in LLM-generated code:\n{code_str}"
            )

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}

    empty_fn = lambda *args, **kwargs: None
    custom_gvars = {
        **gvars,
        "exec": empty_fn,
        "eval": empty_fn,
    }
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        logger.error(f"Error executing LLM-generated code:\n{code_str}")
        raise


# Default LMP configurations for each skill
DEFAULT_LMP_CONFIGS = {
    "parse_query_obj": {
        "prompt_fname": "parse_query_obj_prompt",
        "stop": ["# Query:"],
        "query_prefix": "# Query: ",
        "query_suffix": ".",
        "maintain_session": False,
        "include_context": True,
        "has_return": True,
        "return_val_name": "ret_val",
    },
    "get_affordance_map": {
        "prompt_fname": "get_affordance_map_prompt",
        "stop": ["# Query:"],
        "query_prefix": "# Query: ",
        "query_suffix": ".",
        "maintain_session": False,
        "include_context": False,
        "has_return": True,
        "return_val_name": "ret_val",
    },
    "get_avoidance_map": {
        "prompt_fname": "get_avoidance_map_prompt",
        "stop": ["# Query:"],
        "query_prefix": "# Query: ",
        "query_suffix": ".",
        "maintain_session": False,
        "include_context": False,
        "has_return": True,
        "return_val_name": "ret_val",
    },
    "composer": {
        "prompt_fname": "composer_prompt",
        "stop": ["# Query:"],
        "query_prefix": "# Query: ",
        "query_suffix": ".",
        "maintain_session": True,
        "include_context": True,
        "has_return": True,
        "return_val_name": "ret_val",
    },
    "scene_grounding": {
        # Task 7 Phase 2 — multimodal grounding LMP. Receives an annotated
        # overhead JPEG via `image_bytes=` on the call site. Emits a dict
        # with `blocks_visible` + `ambiguous_resolutions` (no fixtures_state
        # per Phase 0 audit pivot — fixture state comes from scene_obs).
        "prompt_fname": "scene_grounding_prompt",
        "stop": ["# Query:"],
        "query_prefix": "# Query: instruction = ",
        "query_suffix": ", annotated scene.",
        "maintain_session": False,
        "include_context": False,
        "has_return": True,
        "return_val_name": "ret_val",
    },
}


def setup_lmp(config: dict) -> tuple:
    """Set up the LMP hierarchy for CALVIN value map generation.

    Creates CalvinLMPInterface + LMP instances for each skill, wired together
    so the composer can call sub-LMPs (parse_query_obj, get_affordance_map, etc.).

    Args:
        config: Dict with keys: map_size, workspace_bounds_min/max,
                llm_provider, llm_model, llm_temperature, llm_max_tokens,
                cache_dir, load_cache

    Returns:
        (lmps, lmp_interface): Dict of LMP instances and the CalvinLMPInterface
    """
    # Create CALVIN interface
    lmp_interface = CalvinLMPInterface(config)

    # Create LLM backend
    cache = DiskCache(
        cache_dir=config.get("cache_dir", "cache/voxposer_llm"),
        load_cache=config.get("load_cache", True),
    )
    backend = LLMBackend(
        provider=config.get("llm_provider", "anthropic"),
        model=config.get("llm_model", "claude-sonnet-4-20250514"),
        temperature=config.get("llm_temperature", 0),
        max_tokens=config.get("llm_max_tokens", 512),
        cache=cache,
    )

    # Fixed vars: numpy and utility functions
    fixed_vars = {
        "np": np,
    }

    # Variable vars: CalvinLMPInterface methods exposed to LLM code
    variable_vars = {
        k: getattr(lmp_interface, k)
        for k in dir(lmp_interface)
        if callable(getattr(lmp_interface, k)) and not k.startswith("_")
    }

    # Merge user-provided LMP configs with defaults
    lmp_configs = {}
    user_lmp_configs = config.get("lmps", {})
    for name, default_cfg in DEFAULT_LMP_CONFIGS.items():
        merged = {**default_cfg}
        if name in user_lmp_configs:
            merged.update(user_lmp_configs[name])
        lmp_configs[name] = merged

    # Create low-level LMPs (parse_query_obj, get_*_map)
    low_level_names = [n for n in lmp_configs if n not in ["composer", "planner"]]
    low_level_lmps = {
        name: LMP(name, lmp_configs[name], fixed_vars, variable_vars, backend)
        for name in low_level_names
    }
    variable_vars.update(low_level_lmps)

    # Create composer LMP (can call low-level LMPs)
    composer = LMP(
        "composer", lmp_configs["composer"], fixed_vars, variable_vars, backend
    )
    variable_vars["composer"] = composer

    lmps = {
        "composer": composer,
        **low_level_lmps,
    }

    return lmps, lmp_interface


def set_lmp_objects(lmps: dict, objects: list):
    """Set the object context for all LMPs.

    This injects `objects = [...]` into LMPs that use include_context,
    allowing the LLM to know which objects are available.
    """
    for lmp in lmps.values():
        lmp._context = f"objects = {objects}"


# ============================================================================
# Vocab-adherence linter + re-prompt loop (Phase 3a iter 1)
# ============================================================================


def _vocabs() -> tuple:
    """Lazy fetch of the canonical vocab sets (avoids a circular import)."""
    from steering.stage_spec import OBJECT_VOCAB, PRIMITIVE_VOCAB, VALID_STAGE_MODES

    return PRIMITIVE_VOCAB, OBJECT_VOCAB, VALID_STAGE_MODES


def _suggest_object(offender: Any) -> str:
    """Best-fit valid OBJECT_VOCAB entry for an invalid emission."""
    if isinstance(offender, str) and offender in HANDLE_ALIASES:
        return HANDLE_ALIASES[offender]
    _, OBJECT_VOCAB, _ = _vocabs()
    candidate = str(offender) if not isinstance(offender, str) else offender
    matches = get_close_matches(candidate, list(OBJECT_VOCAB), n=1, cutoff=0.0)
    return matches[0] if matches else next(iter(OBJECT_VOCAB))


def _suggest_primitive(offender: Any) -> str:
    """Best-fit valid PRIMITIVE_VOCAB entry for an invalid emission."""
    PRIMITIVE_VOCAB, _, _ = _vocabs()
    candidate = str(offender) if not isinstance(offender, str) else offender
    matches = get_close_matches(
        candidate,
        list(PRIMITIVE_VOCAB),
        n=1,
        cutoff=0.0,
    )
    return matches[0] if matches else next(iter(PRIMITIVE_VOCAB))


def _classify_violations(raw_result: Any) -> list:
    """Walk raw composer output, return the violations needing repair.

    Mirrors `steering.stage_spec._parse_one`'s tuple-shape dispatch but
    returns diagnostic records instead of StageSpecs. An empty list means
    every stage would be accepted by the parser as-is.

    Each violation is a dict with `kind ∈ {primitive, object, shape, length,
    top_level_type}` and the offender's value + a suggested replacement.
    """
    PRIMITIVE_VOCAB, OBJECT_VOCAB, VALID_STAGE_MODES = _vocabs()

    if isinstance(raw_result, tuple):
        raw_stages: list = [raw_result]
    elif isinstance(raw_result, list):
        raw_stages = raw_result
    else:
        return [
            {
                "idx": None,
                "kind": "top_level_type",
                "got": type(raw_result).__name__,
            }
        ]

    violations: list = []
    for i, stage in enumerate(raw_stages):
        if not isinstance(stage, (tuple, list)):
            violations.append(
                {
                    "idx": i,
                    "kind": "shape",
                    "got": type(stage).__name__,
                }
            )
            continue
        n = len(stage)
        if n < 2 or n > 6:
            violations.append({"idx": i, "kind": "length", "got": n})
            continue

        # Extract primitive / object from the same tuple shapes
        # `parse_composer_stages._parse_one` accepts.
        primitive: Any = None
        obj: Any = None
        if n == 4:
            primitive = stage[3]
        elif n == 5:
            if isinstance(stage[2], str) and stage[2] in VALID_STAGE_MODES:
                primitive = stage[3]
                obj = stage[4]
            else:
                primitive = stage[4]
        elif n == 6:
            primitive = stage[4]
            obj = stage[5]

        if primitive is not None and primitive not in PRIMITIVE_VOCAB:
            violations.append(
                {
                    "idx": i,
                    "kind": "primitive",
                    "offender": primitive,
                    "suggested": _suggest_primitive(primitive),
                }
            )
        if obj is not None and obj not in OBJECT_VOCAB:
            violations.append(
                {
                    "idx": i,
                    "kind": "object",
                    "offender": obj,
                    "suggested": _suggest_object(obj),
                }
            )
    return violations


def _build_repair_query(violations: list) -> str:
    """Format a follow-up `# Query:` payload that names the violations + fixes."""
    PRIMITIVE_VOCAB, OBJECT_VOCAB, _ = _vocabs()
    parts = ["The previous response contained invalid values:"]
    for v in violations:
        if v["kind"] == "object":
            parts.append(
                f"Stage {v['idx']}: object={v['offender']!r} is not in "
                f"OBJECT_VOCAB {sorted(OBJECT_VOCAB)}. "
                f"Use {v['suggested']!r} instead."
            )
        elif v["kind"] == "primitive":
            parts.append(
                f"Stage {v['idx']}: primitive={v['offender']!r} is not in "
                f"PRIMITIVE_VOCAB {sorted(PRIMITIVE_VOCAB)}. "
                f"Use {v['suggested']!r} instead."
            )
        elif v["kind"] == "shape":
            parts.append(f"Stage {v['idx']}: expected tuple, got {v['got']}.")
        elif v["kind"] == "length":
            parts.append(
                f"Stage {v['idx']}: tuple length is {v['got']}, valid lengths are 2-6."
            )
        elif v["kind"] == "top_level_type":
            parts.append(
                f"Top-level result type is {v['got']}; "
                f"should be a tuple or list of tuples."
            )
    parts.append(
        "Re-emit the full corrected stage list in the same Python form "
        "(ret_val = ...). Do not change stages that were already valid"
    )
    return " ".join(parts)


class GroundingValidationError(ValueError):
    """Raised when the scene_grounding LMP emits an invalid grounding dict
    (missing required keys, non-OBJECT_VOCAB tokens, etc.). Hard-fails the
    episode same as VocabValidationError / ObjectResolutionError.
    """
    pass


def format_scene_context(grounding: dict) -> str:
    """Pretty-print the grounding dict as a Python comment block.

    Task 7 Phase 2: this block is prepended to the composer's query so it
    appears in the composer's exec history as `# Scene state` context. The
    affordance LMPs read the raw dict via `self._lmp_interface._scene_context`.

    Returns an empty string when `grounding` is None or empty.
    """
    if not grounding:
        return ""
    lines = ["# Scene state (from VLM grounding at episode start):"]
    for k, v in grounding.items():
        if isinstance(v, dict) and v:
            lines.append(f"# {k}:")
            for k2, v2 in v.items():
                lines.append(f"#   {k2}: {v2!r}")
        elif isinstance(v, dict):
            lines.append(f"# {k}: {{}}")
        else:
            lines.append(f"# {k}: {v!r}")
    lines.append("")
    return "\n".join(lines)


def validate_grounding(g: Any) -> list:
    """Vocab-check a grounding dict. Returns list of violation strings."""
    if not isinstance(g, dict):
        return [f"top-level not a dict (got {type(g).__name__})"]
    issues: list = []
    bv = g.get("blocks_visible")
    if not isinstance(bv, dict):
        issues.append("missing or non-dict 'blocks_visible'")
    else:
        valid_buckets = {"table", "drawer_inside", "slider_inside", "held", "absent"}
        for k, v in bv.items():
            if k not in {"red_block", "blue_block", "pink_block"}:
                issues.append(f"blocks_visible.{k}: unknown block color")
            if v not in valid_buckets:
                issues.append(f"blocks_visible.{k}: bucket {v!r} not in {sorted(valid_buckets)}")
    ar = g.get("ambiguous_resolutions")
    if not isinstance(ar, dict):
        issues.append("missing or non-dict 'ambiguous_resolutions'")
    # `slider_accessible_chamber` is optional. Accept 'left' | 'right' | None or
    # missing. Reject other values to surface VLM hallucinations early.
    if "slider_accessible_chamber" in g:
        v = g["slider_accessible_chamber"]
        if v not in (None, "left", "right"):
            issues.append(
                f"slider_accessible_chamber: {v!r} not in ('left', 'right', None)"
            )
    return issues


def compose_with_repair(
    composer: LMP,
    query: str,
    *,
    max_repromptings: int = 2,
    scene_context: Optional[str] = None,
) -> Any:
    """Run the composer with a vocab-adherence linter and re-prompt loop.

    Behaviour:
      1. Call composer(query) → raw result.
      2. Classify violations against the parser's vocab. Empty list → return.
      3. Build a structured repair message, call composer again with it. The
         composer's exec_hist already carries the bad previous output, so the
         LLM sees its own emission plus the fix hints in the new prompt.
      4. Up to `max_repromptings` repair calls (default 2 — 3 LLM calls total).
      5. If still invalid, raise `VocabValidationError` with the violations +
         final raw result. No silent fallback.

    The disk cache keys by prompt text; each repair query has unique text,
    so cache integrity is preserved without manual invalidation.

    Task 7 Phase 2: `scene_context` (an already-formatted comment block from
    `format_scene_context(...)`) is prepended to the query so it appears in
    the composer's prompt + cache key. Pass `None` for text-only composition
    (cache behaves as pre-Task-7).
    """
    composed_query = (
        f"{scene_context}\n{query}" if scene_context else query
    )
    raw_result = composer(composed_query)
    attempts = 0
    while attempts < max_repromptings:
        violations = _classify_violations(raw_result)
        if not violations:
            return raw_result
        attempts += 1
        logger.warning(
            f"Composer vocab violation on attempt {attempts}/"
            f"{max_repromptings + 1}: {len(violations)} issue(s); re-prompting"
        )
        for v in violations:
            offender = v.get("offender", v.get("got"))
            suggested = v.get("suggested", "?")
            logger.warning(
                f"  stage {v.get('idx')}: {v['kind']}={offender!r} -> "
                f"suggested={suggested!r}"
            )
        repair_query = _build_repair_query(violations)
        if scene_context:
            repair_query = f"{scene_context}\n{repair_query}"
        raw_result = composer(repair_query)

    violations = _classify_violations(raw_result)
    if violations:
        raise VocabValidationError(
            f"Composer failed vocab validation after {max_repromptings} "
            f"re-prompt(s). Final violations: {violations}. "
            f"Final raw result: {raw_result!r}"
        )
    return raw_result
