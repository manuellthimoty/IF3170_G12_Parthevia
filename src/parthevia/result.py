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

    def to_dict(self) -> dict[str, Any]:
        params = dict(self.params)
        # Random Restart menyimpan fungsi inner; JSON hanya menyimpan namanya.
        if callable(params.get("inner")):
            inner = params["inner"]
            params["inner"] = getattr(inner, "__name__", type(inner).__name__)
        return {
            "algorithm": self.algorithm,
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "iterations": self.iterations,
            "history": list(self.history),
            "params": params,
            "extra": dict(self.extra),
        }
