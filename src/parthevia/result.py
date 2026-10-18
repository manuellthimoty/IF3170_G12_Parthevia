from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import State


@dataclass
class SearchResult:
    algorithm: str
    initial_state: State
    final_state: State
    initial_objective: float
    final_objective: float
    iterations: int
    history: list[float]
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
