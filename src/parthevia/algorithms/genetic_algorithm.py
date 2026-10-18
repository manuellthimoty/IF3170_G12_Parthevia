from __future__ import annotations

import random
from time import perf_counter

from ..initial_state import random_initial_state
from ..model import AXES, ConstraintChecker, State, objective
from ..neighbors import move_package, rotate_orientation, rotate_package, swap_packages
from ..result import SearchResult


def _select_parent(population: list[State], fitness: list[float], rng: random.Random,
                   tournament_size: int) -> State:
    candidates = [rng.randrange(len(population)) for _ in range(tournament_size)]
    return population[max(candidates, key=lambda index: fitness[index])]


def _crossover(first: State, second: State, checker: ConstraintChecker,
               rng: random.Random, attempts: int) -> State:
    for _ in range(attempts):
        packages = [
            (package if rng.random() < 0.5 else second[package.id]).copy()
            for package in first
        ]
        child = State(first.truck, packages)
        if checker.is_valid(child):
            return child
    # Tolak anak invalid; fallback tetap acak dan tidak mengubah parent.
    return rng.choice((first, second)).copy()


def _mutate(state: State, checker: ConstraintChecker, rng: random.Random, attempts: int) -> State:
    packages = list(state)
    if not packages:
        return state.copy()
    for _ in range(attempts):
        move_type = rng.choice(("swap", "move", "rotate"))
        if move_type == "swap":
            if len(packages) < 2:
                continue
            first, second = rng.sample(packages, 2)
            candidate = swap_packages(state, first.id, second.id)
        else:
            package = rng.choice(packages)
            if move_type == "rotate":
                orientation = rotate_orientation(package.orientation, rng.choice(AXES))
                candidate = rotate_package(state, package.id, orientation)
            else:
                width, length, height = package.effective_dimensions
                nx = max(0, state.truck.width - width + 1)
                ny = max(0, state.truck.length - length + 1)
                nz = max(0, state.truck.height - height + 1)
                count = nx * ny * nz
                # Sampling seragam semua posisi + outside, tanpa membuat list sel.
                index = rng.randrange(count + 1)
                position = None
                if index < count:
                    x, remainder = divmod(index, ny * nz)
                    y, z = divmod(remainder, nz)
                    position = (x, y, z)
                candidate = move_package(state, package.id, position)
        changed = any(candidate[package.id].position != package.position
                      or candidate[package.id].orientation != package.orientation for package in packages)
        if changed and checker.is_valid(candidate):
            return candidate
    return state.copy()


def genetic_algorithm(
    initial_state: State,
    rng: random.Random | None = None,
    population_size: int = 30,
    max_generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    crossover_attempts: int = 10,
    mutation_attempts: int = 40,
    initial_fill_prob: float = 0.7,
    initial_attempts: int = 20,
    seed: int | None = None,
) -> SearchResult:
    """Use input as a problem template; history includes random generation 0."""
    for name, value, minimum in (
        ("population_size", population_size, 2),
        ("max_generations", max_generations, 0),
        ("tournament_size", tournament_size, 1),
        ("crossover_attempts", crossover_attempts, 1),
        ("mutation_attempts", mutation_attempts, 1),
        ("initial_attempts", initial_attempts, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("crossover_rate", crossover_rate), ("mutation_rate", mutation_rate),
                        ("initial_fill_prob", initial_fill_prob)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None")
    if rng is not None and seed is not None:
        raise ValueError("supply rng or seed, not both")
    if rng is not None and not isinstance(rng, random.Random):
        raise ValueError("rng must be a random.Random instance")
    if isinstance(rng, random.SystemRandom):
        raise ValueError("rng must support reproducible state; use random.Random")

    start = perf_counter()
    rng_state = rng.getstate() if rng is not None else None
    if rng is None:
        seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
        rng = random.Random(seed)
    checker = ConstraintChecker(initial_state.truck)
    population = [
        random_initial_state(initial_state.truck, initial_state, rng,
                             fill_prob=initial_fill_prob, attempts=initial_attempts)
        for _ in range(population_size)
    ]
    history: list[float] = []
    mean_history: list[float] = []
    initial: State | None = None
    best: State | None = None
    best_value = float("-inf")
    for generation in range(max_generations + 1):
        fitness = [objective(state, checker) for state in population]
        best_index = max(range(population_size), key=lambda index: fitness[index])
        generation_best = fitness[best_index]
        history.append(generation_best)
        mean_history.append(sum(fitness) / population_size)
        if generation == 0:
            initial = population[best_index].copy()
        if best is None or generation_best > best_value:
            best = population[best_index].copy()
            best_value = generation_best
        if generation == max_generations:
            break

        # Satu elite tidak ikut crossover/mutasi agar fitness maksimum tidak turun.
        next_population = [population[best_index].copy()]
        for _ in range(population_size - 1):
            first = _select_parent(population, fitness, rng, tournament_size)
            if rng.random() < crossover_rate:
                second = _select_parent(population, fitness, rng, tournament_size)
                child = _crossover(first, second, checker, rng, crossover_attempts)
            else:
                child = first.copy()
            if rng.random() < mutation_rate:
                child = _mutate(child, checker, rng, mutation_attempts)
            next_population.append(child)
        population = next_population

    assert initial is not None and best is not None
    return SearchResult(
        algorithm="genetic_algorithm",
        initial_state=initial,
        final_state=best.copy(),
        initial_objective=history[0],
        final_objective=best_value,
        iterations=max_generations,
        history=history,
        params={
            "population_size": population_size,
            "max_generations": max_generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "elite_count": 1,
            "crossover_attempts": crossover_attempts,
            "mutation_attempts": mutation_attempts,
            "initial_fill_prob": initial_fill_prob,
            "initial_attempts": initial_attempts,
            "seed": seed,
            "rng_state": rng_state,
            "initial_selection": "first_max_fitness_generation_0",
        },
        extra={
            "mean_history": mean_history,
            "history_includes_generation_0": True,
            "duration_seconds": perf_counter() - start,
        },
    )
