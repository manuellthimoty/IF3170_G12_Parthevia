from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

from ..model import ConstraintChecker, State, objective
from ..neighbors import enumerate_neighbors

NeighborFunction = Callable[[State], Iterable[State]]
ObjectiveFunction = Callable[[State], float]


@dataclass
class HillClimbingResult:
    initial_state: State
    final_state: State
    history: list[float]
    iterations: int
    duration_seconds: float
    stop_reason: str
    sideways_moves: int = 0

    @property
    def final_value(self) -> float:
        return self.history[-1]

    def to_dict(self) -> dict:
        return {
            "initialState": self.initial_state.to_dict(),
            "finalState": self.final_state.to_dict(),
            "objectiveHistory": self.history,
            "iterations": self.iterations,
            "durationSeconds": self.duration_seconds,
            "stopReason": self.stop_reason,
            "sidewaysMoves": self.sideways_moves,
        }


class SteepestAscentHillClimbing:
    def __init__(
        self,
        objective_fn: ObjectiveFunction | None = None,
        neighbor_fn: NeighborFunction | None = None,
        checker: ConstraintChecker | None = None,
        max_iterations: int = 1_000,
    ):
        self._checker = checker
        self._objective = objective_fn or (lambda state: objective(state, checker))
        self._neighbors = neighbor_fn or (lambda state: enumerate_neighbors(state, checker))
        self.max_iterations = max_iterations

    def run(self, initial_state: State) -> HillClimbingResult:
        start = perf_counter()
        initial = initial_state.copy()
        current = initial.copy()
        current_value = self._objective(current)
        history = [current_value]
        iterations = 0
        stop_reason = "local_optimum"

        while iterations < self.max_iterations:
            neighbors = list(self._neighbors(current))
            if not neighbors:
                stop_reason = "no_valid_neighbors"
                break
            best = max(neighbors, key=self._objective)
            best_value = self._objective(best)
            if best_value <= current_value:
                stop_reason = "local_optimum"
                break
            current = best.copy()
            current_value = best_value
            history.append(current_value)
            iterations += 1
        else:
            stop_reason = "max_iterations"

        return HillClimbingResult(
            initial_state=initial,
            final_state=current,
            history=history,
            iterations=iterations,
            duration_seconds=perf_counter() - start,
            stop_reason=stop_reason,
        )


class SidewaysHillClimbing(SteepestAscentHillClimbing):
    def __init__(self, *args, max_sideways: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sideways = max_sideways

    def run(self, initial_state: State) -> HillClimbingResult:
        start = perf_counter()
        initial = initial_state.copy()
        current = initial.copy()
        current_value = self._objective(current)
        history = [current_value]
        iterations = 0
        sideways = 0
        stop_reason = "local_optimum"
        while iterations < self.max_iterations:
            neighbors = list(self._neighbors(current))
            if not neighbors:
                stop_reason = "no_valid_neighbors"
                break
            best = max(neighbors, key=self._objective)
            best_value = self._objective(best)
            if best_value < current_value:
                stop_reason = "local_optimum"
                break
            if best_value == current_value:
                if sideways >= self.max_sideways:
                    stop_reason = "max_sideways"
                    break
                sideways += 1
            current = best.copy()
            current_value = best_value
            history.append(current_value)
            iterations += 1
        else:
            stop_reason = "max_iterations"

        return HillClimbingResult(
            initial_state=initial,
            final_state=current,
            history=history,
            iterations=iterations,
            duration_seconds=perf_counter() - start,
            stop_reason=stop_reason,
            sideways_moves=sideways,
        )


SteepestAscentHC = SteepestAscentHillClimbing
HillClimbing = SteepestAscentHillClimbing
