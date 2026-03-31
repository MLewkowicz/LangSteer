"""Utility classes for VoxPoser value map generation.

Ported from VoxPoser/src/utils.py with minimal changes.
"""

import os
import numpy as np


def load_prompt(prompt_fname: str) -> str:
    """Load a prompt template from the prompts directory."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if '/' in prompt_fname:
        parts = prompt_fname.split('/')
        full_path = os.path.join(curr_dir, 'prompts', *parts)
    else:
        full_path = os.path.join(curr_dir, 'prompts', prompt_fname)
    with open(full_path, 'r') as f:
        return f.read().strip()


def normalize_vector(x, eps=1e-6):
    """Normalize a vector to unit length."""
    x = np.asarray(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return np.zeros_like(x) if norm < eps else (x / norm)
    elif x.ndim == 2:
        norm = np.linalg.norm(x, axis=1)
        normalized = np.zeros_like(x)
        normalized[norm > eps] = x[norm > eps] / norm[norm > eps][:, None]
        return normalized
    return x


def normalize_map(voxel_map):
    """Normalize voxel map values to [0, 1] without producing nan."""
    denom = voxel_map.max() - voxel_map.min()
    if denom == 0:
        return voxel_map
    return (voxel_map - voxel_map.min()) / denom


class Observation(dict):
    """Attribute-access dict for object observations.

    Allows both dict-style (obs['position']) and attribute-style (obs.position) access.
    Used by LLM-generated code to access object properties like .position, .aabb, .name.
    """

    def __init__(self, obs_dict):
        super().__init__(obs_dict)
        self.obs_dict = obs_dict

    def __getattr__(self, key):
        if key == 'obs_dict':
            return super().__getattribute__('obs_dict')
        return self.obs_dict[key]

    def __getitem__(self, key):
        return self.obs_dict[key]

    def __getstate__(self):
        return self.obs_dict

    def __setstate__(self, state):
        self.obs_dict = state


class DynamicObservation:
    """Lazy-eval wrapper that uses latest sensor readings.

    Wraps a callable that returns an Observation dict. Each access
    evaluates the function to get the latest state.
    """

    def __init__(self, func):
        assert callable(func) and not isinstance(func, dict), \
            'func must be callable and not a dict'
        self.func = func

    def __get__(self, key):
        evaluated = self.func()
        if isinstance(evaluated[key], np.ndarray):
            return evaluated[key].copy()
        return evaluated[key]

    def __getattr__(self, key):
        if key == 'func':
            return super().__getattribute__('func')
        return self.__get__(key)

    def __getitem__(self, key):
        return self.__get__(key)

    def __call__(self):
        static_obs = self.func()
        if not isinstance(static_obs, Observation):
            static_obs = Observation(static_obs)
        return static_obs


class IterableDynamicObservation:
    """Lazy-eval wrapper for lists of observations.

    Wraps a callable returning a list. Each element access evaluates
    the function to get the latest state.
    """

    def __init__(self, func):
        assert callable(func), 'func must be callable'
        self.func = func
        self._validate_func_output()

    def _validate_func_output(self):
        evaluated = self.func()
        assert isinstance(evaluated, list), 'func must evaluate to a list'

    def __getitem__(self, index):
        def helper():
            evaluated = self.func()
            return evaluated[index]
        return helper

    def __len__(self):
        return len(self.func())

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __call__(self):
        return self.func()


def _process_llm_index(indices, array_shape):
    """Process indices from LLM-generated code for safe array access.

    Handles non-integer indexing, negative indexing, and out-of-bounds
    indices that LLMs sometimes generate.
    """
    if isinstance(indices, (int, np.int64, np.int32, np.int16, np.int8)):
        processed = indices if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, (float, np.float64, np.float32, np.float16)):
        processed = np.round(indices).astype(int) if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, slice):
        start, stop, step = indices.start, indices.stop, indices.step
        if start is not None:
            start = int(np.round(start))
        if stop is not None:
            stop = int(np.round(stop))
        if step is not None:
            step = int(np.round(step))
        if (start is not None and start < 0) and (stop is not None):
            if stop >= 0:
                processed = slice(0, stop, step)
            else:
                processed = slice(0, 0, step)
        else:
            processed = slice(start, stop, step)
    elif isinstance(indices, (tuple, list)):
        processed = tuple(
            _process_llm_index(idx, (array_shape[i],)) for i, idx in enumerate(indices)
        )
    elif isinstance(indices, np.ndarray):
        processed = _process_llm_index(indices.tolist(), array_shape)
    else:
        raise TypeError(f"Indexing type {type(indices)} not supported")
    return processed


class VoxelIndexingWrapper:
    """Numpy array wrapper with LLM-safe indexing.

    Processes indices through _process_llm_index to handle float indices,
    negative indexing, and other edge cases from LLM-generated code.
    Behaves like a numpy array for all other operations.
    """

    def __init__(self, array):
        self.array = array

    def __getitem__(self, idx):
        return self.array[_process_llm_index(idx, tuple(self.array.shape))]

    def __setitem__(self, idx, value):
        self.array[_process_llm_index(idx, tuple(self.array.shape))] = value

    def __repr__(self):
        return self.array.__repr__()

    def __str__(self):
        return self.array.__str__()

    # Comparison operators
    def __eq__(self, other): return self.array == other
    def __ne__(self, other): return self.array != other
    def __lt__(self, other): return self.array < other
    def __le__(self, other): return self.array <= other
    def __gt__(self, other): return self.array > other
    def __ge__(self, other): return self.array >= other

    # Arithmetic operators
    def __add__(self, other): return self.array + other
    def __sub__(self, other): return self.array - other
    def __mul__(self, other): return self.array * other
    def __truediv__(self, other): return self.array / other
    def __floordiv__(self, other): return self.array // other
    def __mod__(self, other): return self.array % other
    def __pow__(self, other): return self.array ** other

    # Reverse arithmetic operators
    def __radd__(self, other): return other + self.array
    def __rsub__(self, other): return other - self.array
    def __rmul__(self, other): return other * self.array
    def __rtruediv__(self, other): return other / self.array
    def __rfloordiv__(self, other): return other // self.array
    def __rmod__(self, other): return other % self.array
    def __rpow__(self, other): return other ** self.array

    def __getattr__(self, name):
        if name == 'array':
            return super().__getattribute__('array')
        return getattr(self.array, name)
