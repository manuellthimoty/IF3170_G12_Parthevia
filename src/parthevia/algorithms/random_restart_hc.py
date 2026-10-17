from __future__ import annotations

import random
from typing import Any, Callable, Mapping

from ..initial_state import random_initial_state
from ..model import ConstraintChecker, State, objective
from ..result import SearchResult
from .stochastic_hc import stochastic_hill_climbing

HillClimber = Callable[..., SearchResult]


def random_restart_hill_climbing(
    initial_state: State,
    rng: random.Random | None = None,
    max_restart: int = 10,
    inner: HillClimber = stochastic_hill_climbing,
    inner_params: Mapping[str, Any] | None = None,
) -> SearchResult:
    rng = rng or random.Random()
    inner_params = dict(inner_params or {})
    packages = list(initial_state)

    best: SearchResult | None = None
    history: list[float] = []
    iterations_per_restart: list[int] = []
    objective_per_restart: list[float] = []
    restart_starts: list[int] = []
    restarts = 0
    start = initial_state
    while True:
        result = inner(start, rng, **inner_params)
        restart_starts.append(len(history))
        history.extend(result.history)
        iterations_per_restart.append(result.iterations)
        objective_per_restart.append(result.final_objective)
        if best is None or result.final_objective > best.final_objective:
            best = result
        if restarts >= max_restart:
            break
        restarts += 1
        start = random_initial_state(initial_state.truck, packages, rng)

    assert best is not None
    return SearchResult(
        algorithm="random_restart_hill_climbing",
        initial_state=initial_state,
        final_state=best.final_state,
        initial_objective=objective(initial_state, ConstraintChecker(initial_state.truck)),
        final_objective=best.final_objective,
        iterations=sum(iterations_per_restart),
        history=history,
        params={"max_restart": max_restart, "inner": inner, "inner_params": inner_params},
        extra={
            "restarts": restarts,
            "iterations_per_restart": iterations_per_restart,
            "objective_per_restart": objective_per_restart,
            "restart_starts": restart_starts,
            "best_restart": objective_per_restart.index(best.final_objective),
        },
    )
