"""Registry of named Risk environments.

Gymnasium's ``make()`` expects a single-agent ``Env``. Risk is a PettingZoo
AEC environment, so this module keeps its own registry and factory.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

env_list: list[str] = []
_REGISTRY: dict[str, str] = {}


def register(id: str, entry_point: str, reward_threshold: float = 0.95) -> None:
    """Register a Risk environment class under ``id``.

    ``entry_point`` is ``"module.path:ClassName"``, matching the historical
    Gym registration string.
    """
    del reward_threshold  # kept for call-site compatibility with the old API
    assert id.startswith("Risk-")
    assert id not in env_list
    _REGISTRY[id] = entry_point
    env_list.append(id)


def _load_entry_point(entry_point: str) -> type:
    module_name, _, attr = entry_point.partition(":")
    if not attr:
        raise ValueError(f"Invalid entry point (expected 'module:Class'): {entry_point}")
    module = import_module(module_name)
    return getattr(module, attr)


def make(id: str, **kwargs: Any):
    """Instantiate a registered Risk environment.

    Parameters
    ----------
    id:
        One of the ids in :data:`env_list`, e.g. ``"Risk-Normal-6-v0"``.
    **kwargs:
        Forwarded to the environment constructor (overrides the class defaults).
    """
    # Importing envs populates the registry (idempotent).
    import pz_risk.envs  # noqa: F401

    if id not in _REGISTRY:
        known = ", ".join(env_list) or "<none registered>"
        raise KeyError(f"Unknown environment {id!r}. Known ids: {known}")
    cls = _load_entry_point(_REGISTRY[id])
    return cls(**kwargs)
