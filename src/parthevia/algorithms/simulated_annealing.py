from __future__ import annotations

import math
import random

from ..model import ConstraintChecker, State, objective
from ..neighbors import random_neighbor
from ..result import SearchResult


def estimate_initial_temperature(
    state: State,
    rng: random.Random,
    samples: int = 200,
    initial_accept: float = 0.8,
) -> float:
    checker = ConstraintChecker(state.truck)
    current_value = state.total_value
    losses = []
    for _ in range(samples):
        neighbor = random_neighbor(state, checker, rng)
        if neighbor is not None and neighbor.total_value < current_value:
            losses.append(current_value - neighbor.total_value)
    if not losses:
        values = [package.value for package in state]
        return max(1.0, sum(values) / max(1, len(values)))
    return (sum(losses) / len(losses)) / -math.log(initial_accept)


def simulated_annealing(
    initial_state: State,
    rng: random.Random | None = None,
    initial_temperature: float | None = None,
    cooling_rate: float = 0.999,
    min_temperature: float = 1e-3,
    max_iterations: int = 20000,
    stuck_window: int = 200,
    neighbor_attempts: int = 40,
) -> SearchResult:
    rng = rng or random.Random()
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be in (0, 1)")
    checker = ConstraintChecker(initial_state.truck)
    t0 = initial_temperature if initial_temperature is not None else estimate_initial_temperature(initial_state, rng)

    current = initial_state
    current_value = current.total_value
    best, best_value = current, current_value

    history = [current_value]
    best_history = [best_value]
    temperatures = [t0]
    acceptance = []  # isinya kurleb [iterasi, T, dE, e^(dE/T), diterima?] untuk setiap dE <= 0
    accepted_better = accepted_worse = rejected_worse = sideways = no_neighbor = 0
    stuck_count, since_improve, stuck_events = 0, 0, []

    iteration = 0
    temperature = t0
    while iteration < max_iterations:
        temperature = t0*cooling_rate**(iteration + 1)
        if temperature<min_temperature:
            break
        iteration +=1

        neighbor = random_neighbor(current, checker, rng, attempts=neighbor_attempts)
        if neighbor is None:
            no_neighbor +=1
        else:
            delta = neighbor.total_value - current_value
            if delta > 0:
                accepted = True
                accepted_better +=1
            else:
                probability = math.exp(delta / temperature)
                accepted = rng.random() < probability
                if delta ==0:
                    sideways+=accepted
                else:
                    acceptance.append([iteration, temperature, delta, probability, accepted])
                    if accepted:
                        accepted_worse+= 1
                    else:
                        rejected_worse += 1
            if accepted:
                current, current_value = neighbor, neighbor.total_value

        if current_value > best_value:
            best, best_value = current, current_value
            since_improve = 0
        else:
            since_improve += 1
            if since_improve == stuck_window:
                stuck_count += 1
                stuck_events.append(iteration)

        history.append(current_value)
        best_history.append(best_value)
        temperatures.append(temperature)

    return SearchResult(
        algorithm="simulated_annealing",
        initial_state=initial_state,
        final_state=best,
        initial_objective=objective(initial_state, checker),
        final_objective=objective(best, checker),
        iterations=iteration,
        history=history,
        params={
            "initial_temperature": t0,
            "cooling_rate": cooling_rate,
            "min_temperature": min_temperature,
            "max_iterations": max_iterations,
            "stuck_window": stuck_window,
            "neighbor_attempts": neighbor_attempts,
        },
        extra={
            "best_history": best_history,
            "temperatures": temperatures,
            "acceptance": acceptance,
            "final_temperature": temperature,
            "last_state_objective": current_value,
            "stuck_count": stuck_count,
            "stuck_events": stuck_events,
            "accepted_better": accepted_better,
            "accepted_worse": accepted_worse,
            "rejected_worse": rejected_worse,
            "sideways": sideways,
            "no_neighbor": no_neighbor,
        },
    )
