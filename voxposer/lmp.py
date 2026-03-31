"""Language Model Program (LMP) engine for VoxPoser value map generation.

Modernized from VoxPoser/src/LMP.py with provider-agnostic LLM backends
(Anthropic Claude and OpenAI GPT). Uses in-context learning prompts to
generate Python code for 3D value map construction.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from voxposer.calvin_interface import CalvinLMPInterface
from voxposer.llm_cache import DiskCache
from voxposer.utils import (
    DynamicObservation,
    IterableDynamicObservation,
    Observation,
    load_prompt,
)

logger = logging.getLogger(__name__)

# System prompt for the LLM (shared across providers)
SYSTEM_PROMPT = (
    "You are a helpful assistant that writes Python code to control a robot arm "
    "in a tabletop manipulation environment (CALVIN benchmark). Complete the code "
    "when given a new query. Follow the patterns in the context code. Be thorough "
    "and thoughtful. Do not include import statements. Do not repeat the query. "
    "Do not provide text explanations (comments in code are okay). "
    "Note: x is left(-) to right(+), y is back(-) to front(+), z is bottom to top."
)


class LLMBackend:
    """Provider-agnostic LLM API wrapper."""

    def __init__(self, provider: str, model: str, temperature: float,
                 max_tokens: int, cache: DiskCache):
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cache = cache
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self._provider == 'anthropic':
            import anthropic
            self._client = anthropic.Anthropic()
        elif self._provider == 'openai':
            import openai
            self._client = openai.OpenAI()
        else:
            raise ValueError(f"Unknown LLM provider: {self._provider}")
        return self._client

    def generate(self, prompt: str, stop: list) -> str:
        """Generate code completion from the LLM, with caching."""
        cache_key = {
            'provider': self._provider,
            'model': self._model,
            'prompt': prompt,
            'temperature': self._temperature,
            'max_tokens': self._max_tokens,
        }

        if cache_key in self._cache:
            logger.debug("Using cached LLM response")
            return self._cache[cache_key]

        client = self._get_client()
        start_time = time.time()

        # Split prompt into context and query for chat-style APIs
        parts = prompt.rsplit('# Query:', 1)
        if len(parts) == 2:
            context = parts[0].strip()
            query = '# Query:' + parts[1]
        else:
            context = prompt
            query = ""

        # Retry with exponential backoff
        for attempt in range(5):
            try:
                if self._provider == 'anthropic':
                    result = self._call_anthropic(client, context, query, stop)
                else:
                    result = self._call_openai(client, context, query, stop)
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM API error (attempt {attempt + 1}): {e}. "
                               f"Retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError("LLM API failed after 5 attempts")

        elapsed = time.time() - start_time
        logger.info(f"LLM API call ({self._provider}/{self._model}) "
                     f"took {elapsed:.2f}s")

        self._cache[cache_key] = result
        return result

    def _call_anthropic(self, client, context: str, query: str,
                        stop: list) -> str:
        messages = [
            {"role": "user", "content": (
                f"I will give you context code, then a query to complete.\n\n"
                f"Context:\n```\n{context}\n```\n\n"
                f"Complete this:\n{query}"
            )},
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
        text = text.replace('```python', '').replace('```', '').strip()
        return text

    def _call_openai(self, client, context: str, query: str,
                     stop: list) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"I will give you context code, then a query to complete.\n\n"
                f"Context:\n```\n{context}\n```\n\n"
                f"Complete this:\n{query}"
            )},
        ]
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stop=stop if stop else None,
            messages=messages,
        )
        text = response.choices[0].message.content.strip()
        text = text.replace('```python', '').replace('```', '').strip()
        return text


class LMP:
    """Language Model Program: generates and executes Python code from LLM.

    Each LMP instance corresponds to one skill (composer, get_affordance_map,
    parse_query_obj, etc.). It loads a prompt template, sends queries to the
    LLM, and safely executes the returned code.
    """

    def __init__(self, name: str, cfg: dict, fixed_vars: dict,
                 variable_vars: dict, backend: LLMBackend, env_name: str = 'calvin'):
        self._name = name
        self._cfg = cfg
        self._backend = backend
        self._base_prompt = load_prompt(f"{env_name}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg.get('stop', ['# Query:']))
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query: str):
        """Build the full prompt from template + session history + query."""
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = \
                f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''

        prompt = self._base_prompt.replace(
            '{variable_vars_imports}', variable_vars_imports_str
        )

        if self._cfg.get('maintain_session') and self.exec_hist:
            prompt += f'\n{self.exec_hist}'

        prompt += '\n'

        if self._cfg.get('include_context') and self._context is not None:
            prompt += f'\n{self._context}'

        query_prefix = self._cfg.get('query_prefix', '# Query: ')
        query_suffix = self._cfg.get('query_suffix', '.')
        user_query = f'{query_prefix}{query}{query_suffix}'
        prompt += f'\n{user_query}'

        return prompt, user_query

    def __call__(self, query: str, **kwargs):
        """Generate code for the query, execute it, and return the result."""
        prompt, user_query = self.build_prompt(query)

        code_str = self._backend.generate(prompt, self._stop_tokens)

        if self._cfg.get('include_context') and self._context is not None:
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        logger.info(
            f'[LMP "{self._name}"] generated code:\n{to_log}'
        )

        gvars = {**self._fixed_vars, **self._variable_vars}
        lvars = kwargs

        # Non-composer LMPs return functions for lazy evaluation
        if self._name not in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg.get('maintain_session'):
            self._variable_vars.update(lvars)

        if self._cfg.get('has_return'):
            return_val_name = self._cfg.get('return_val_name', 'ret_val')
            if self._name == 'parse_query_obj':
                try:
                    return IterableDynamicObservation(lvars[return_val_name])
                except (AssertionError, AssertionError):
                    return DynamicObservation(lvars[return_val_name])
            return lvars[return_val_name]


def exec_safe(code_str: str, gvars: Optional[dict] = None,
              lvars: Optional[dict] = None):
    """Execute code string in a sandboxed environment.

    Bans import statements and dunder access for safety.
    """
    banned_phrases = ['import', '__']
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
        'exec': empty_fn,
        'eval': empty_fn,
    }
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        logger.error(f'Error executing LLM-generated code:\n{code_str}')
        raise


# Default LMP configurations for each skill
DEFAULT_LMP_CONFIGS = {
    'parse_query_obj': {
        'prompt_fname': 'parse_query_obj_prompt',
        'stop': ['# Query:'],
        'query_prefix': '# Query: ',
        'query_suffix': '.',
        'maintain_session': False,
        'include_context': True,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'get_affordance_map': {
        'prompt_fname': 'get_affordance_map_prompt',
        'stop': ['# Query:'],
        'query_prefix': '# Query: ',
        'query_suffix': '.',
        'maintain_session': False,
        'include_context': False,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'get_avoidance_map': {
        'prompt_fname': 'get_avoidance_map_prompt',
        'stop': ['# Query:'],
        'query_prefix': '# Query: ',
        'query_suffix': '.',
        'maintain_session': False,
        'include_context': False,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'get_gripper_map': {
        'prompt_fname': 'get_gripper_map_prompt',
        'stop': ['# Query:'],
        'query_prefix': '# Query: ',
        'query_suffix': '.',
        'maintain_session': False,
        'include_context': False,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'composer': {
        'prompt_fname': 'composer_prompt',
        'stop': ['# Query:'],
        'query_prefix': '# Query: ',
        'query_suffix': '.',
        'maintain_session': True,
        'include_context': True,
        'has_return': True,
        'return_val_name': 'ret_val',
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
        cache_dir=config.get('cache_dir', 'cache/voxposer_llm'),
        load_cache=config.get('load_cache', True),
    )
    backend = LLMBackend(
        provider=config.get('llm_provider', 'anthropic'),
        model=config.get('llm_model', 'claude-sonnet-4-20250514'),
        temperature=config.get('llm_temperature', 0),
        max_tokens=config.get('llm_max_tokens', 512),
        cache=cache,
    )

    # Fixed vars: numpy and utility functions
    fixed_vars = {
        'np': np,
    }

    # Variable vars: CalvinLMPInterface methods exposed to LLM code
    variable_vars = {
        k: getattr(lmp_interface, k)
        for k in dir(lmp_interface)
        if callable(getattr(lmp_interface, k)) and not k.startswith('_')
    }

    # Merge user-provided LMP configs with defaults
    lmp_configs = {}
    user_lmp_configs = config.get('lmps', {})
    for name, default_cfg in DEFAULT_LMP_CONFIGS.items():
        merged = {**default_cfg}
        if name in user_lmp_configs:
            merged.update(user_lmp_configs[name])
        lmp_configs[name] = merged

    # Create low-level LMPs (parse_query_obj, get_*_map)
    low_level_names = [n for n in lmp_configs if n not in ['composer', 'planner']]
    low_level_lmps = {
        name: LMP(name, lmp_configs[name], fixed_vars, variable_vars, backend)
        for name in low_level_names
    }
    variable_vars.update(low_level_lmps)

    # Create composer LMP (can call low-level LMPs)
    composer = LMP('composer', lmp_configs['composer'], fixed_vars, variable_vars, backend)
    variable_vars['composer'] = composer

    lmps = {
        'composer': composer,
        **low_level_lmps,
    }

    return lmps, lmp_interface


def set_lmp_objects(lmps: dict, objects: list):
    """Set the object context for all LMPs.

    This injects `objects = [...]` into LMPs that use include_context,
    allowing the LLM to know which objects are available.
    """
    for lmp in lmps.values():
        lmp._context = f'objects = {objects}'
