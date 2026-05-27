"""Utilities for loading and executing VLM-generated guidance functions.

Ported and trimmed from the VLS repo (vls/utils/guidance_utils.py).
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def exec_safe(
    code_str: str, gvars: dict | None = None, lvars: dict | None = None
) -> None:
    """Execute code string with minimal safety restrictions.

    Bans dunder-method names to prevent most code-injection patterns while
    still allowing the import statements that VLM-generated functions need.
    """
    for phrase in ["__"]:
        assert phrase not in code_str, f"Banned phrase '{phrase}' found in code"

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}

    def _noop(*a, **k):
        return None

    merged = {**gvars, "exec": _noop, "eval": _noop}
    try:
        exec(code_str, merged, lvars)
    except Exception:
        logger.error(f"Error executing generated code:\n{code_str}")
        raise


def _numpy_to_torch(code: str) -> str:
    """Replace common numpy calls with torch equivalents in generated code."""
    code = code.replace("np.array(", "torch.tensor(")
    code = code.replace("numpy.array(", "torch.tensor(")
    replacements = {
        "np.zeros": "torch.zeros",
        "np.ones": "torch.ones",
        "np.linalg.norm": "torch.norm",
        "np.sum": "torch.sum",
        "np.mean": "torch.mean",
        "np.max": "torch.max",
        "np.min": "torch.min",
        "np.sqrt": "torch.sqrt",
        "np.abs": "torch.abs",
        "np.exp": "torch.exp",
        "np.log": "torch.log",
    }
    for np_fn, torch_fn in replacements.items():
        code = code.replace(np_fn, torch_fn)
    return code


def load_functions_from_string(code: str, validate: bool = True) -> list[Callable]:
    """Parse guidance functions from a code string, returning callables.

    Args:
        code: Python source containing one or more ``stageN_guidance`` functions.
        validate: Run a quick forward-pass check with dummy inputs.

    Returns:
        List of wrapped, device-aware callables, one per parsed function.
    """
    code = _numpy_to_torch(code)

    gvars: dict = {"torch": torch}
    lvars: dict = {}
    exec_safe(code, gvars=gvars, lvars=lvars)

    fns = [wrap_with_device(fn) for fn in lvars.values() if callable(fn)]

    if validate and fns:
        _validate(fns)

    return fns


def wrap_with_device(fn: Callable) -> Callable:
    """Wrap a guidance function so tensor literals land on the right device.

    Monkey-patches the ``torch`` reference in the function's ``__globals__``
    with a thin proxy that injects ``device`` from the live ``trajectory_3d``
    argument into every tensor-creation call.
    """

    def wrapped(keypoints: torch.Tensor, trajectory_3d: torch.Tensor) -> torch.Tensor:
        device = trajectory_3d.device

        if keypoints.device != device:
            keypoints = keypoints.to(device)

        if not hasattr(fn, "__globals__"):
            return fn(keypoints, trajectory_3d)

        class _TorchProxy:
            def tensor(self, *a, **kw):
                kw.setdefault("device", device)
                return torch.tensor(*a, **kw)

            def zeros(self, *a, **kw):
                kw.setdefault("device", device)
                return torch.zeros(*a, **kw)

            def ones(self, *a, **kw):
                kw.setdefault("device", device)
                return torch.ones(*a, **kw)

            def arange(self, *a, **kw):
                kw.setdefault("device", device)
                return torch.arange(*a, **kw)

            def __getattr__(self, name: str):
                return getattr(torch, name)

        orig = fn.__globals__.get("torch", torch)
        fn.__globals__["torch"] = _TorchProxy()
        try:
            result = fn(keypoints, trajectory_3d)
            if (
                result is not None
                and hasattr(result, "device")
                and result.device != device
            ):
                result = result.to(device)
            return result
        finally:
            fn.__globals__["torch"] = orig

    return wrapped


def _validate(fns: list[Callable]) -> None:
    dummy_kp = torch.randn(10, 3)
    dummy_traj = torch.randn(1, 10, 3)
    for i, fn in enumerate(fns):
        try:
            out = fn(dummy_kp, dummy_traj)
            if out is None:
                raise ValueError(f"function {i} returned None")
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"function {i} returned {type(out)}, expected Tensor")
        except Exception as e:
            raise ValueError(f"Guidance function {i} failed validation: {e}") from e
