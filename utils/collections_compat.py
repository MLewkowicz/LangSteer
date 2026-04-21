"""
Restore collections ABC aliases removed in Python 3.10+.

Legacy code does ``from collections import Mapping``; those names live in
``collections.abc`` only. Import this module (or load it via importlib) before
importing affected third-party packages (e.g. SageMaker SDK v3 on older wheels).
"""
from __future__ import annotations

import collections
import collections.abc

_ALIASES = (
    ("Mapping", collections.abc.Mapping),
    ("MutableMapping", collections.abc.MutableMapping),
    ("MutableSet", collections.abc.MutableSet),
    ("Sequence", collections.abc.Sequence),
    ("MutableSequence", collections.abc.MutableSequence),
    ("Set", collections.abc.Set),
    ("ItemsView", collections.abc.ItemsView),
    ("KeysView", collections.abc.KeysView),
    ("ValuesView", collections.abc.ValuesView),
    ("Iterable", collections.abc.Iterable),
    ("Iterator", collections.abc.Iterator),
    ("Callable", collections.abc.Callable),
)

for _name, _value in _ALIASES:
    if not hasattr(collections, _name):
        setattr(collections, _name, _value)
