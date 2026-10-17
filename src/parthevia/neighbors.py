from __future__ import annotations

import itertools
import random
from typing import Iterable

from .model import AXES, ConstraintChecker, Package, State
ORIENTATIONS = tuple(itertools.permutations(AXES))
def _signature(state: State) -> tuple:
    return tuple(
        (package.id, package.position, package.orientation)
        for package in state
    )

def swap_packages(state: State, first_id: str, second_id: str) -> State:
    if first_id == second_id:
        return state.copy()
    first, second = state[first_id], state[second_id]
    result = state.copy()
    first_copy, second_copy = first.copy(), second.copy()
    first_copy.position, second_copy.position = second.position, first.position
    result.packages[first_id] = first_copy
    result.packages[second_id] = second_copy
    return result

def move_package(state: State, package_id: str, position: tuple[int, int, int] | None) -> State:
    result = state.copy()
    result.packages[package_id].position = position
    return result

def rotate_orientation(orientation: tuple[str, str, str], axis: str) -> tuple[str, str, str]:
    first, second = (item for item in AXES if item != axis)
    swap = {first: second, second: first}
    return tuple(swap.get(item, item) for item in orientation)

def rotate_package(state: State, package_id: str, orientation: tuple[str, str, str]) -> State:
    result = state.copy()
    result.packages[package_id].orientation = orientation
    return result

def _positions_for(package: Package, state: State) -> Iterable[tuple[int, int, int] | None]:
    width, length, height = package.effective_dimensions
    for x in range(0, state.truck.width - width + 1):
        for y in range(0, state.truck.length - length + 1):
            for z in range(0, state.truck.height - height + 1):
                yield (x, y, z)
    yield None

def enumerate_neighbors(
    state: State,
    checker: ConstraintChecker | None = None,
    max_neighbors: int | None = None,
) -> list[State]:
    checker = checker or ConstraintChecker(state.truck)
    result: list[State] = []
    seen = {_signature(state)}

    def add(candidate: State) -> bool:
        signature = _signature(candidate)
        if signature in seen or not checker.is_valid(candidate):
            return False
        seen.add(signature)
        result.append(candidate)
        return max_neighbors is not None and len(result) >= max_neighbors

    packages = list(state)
    for first_index, first in enumerate(packages):
        for second in packages[first_index + 1 :]:
            if add(swap_packages(state, first.id, second.id)):
                return result

    for package in packages:
        for position in _positions_for(package, state):
            if position == package.position:
                continue
            if add(move_package(state, package.id, position)):
                return result

    for package in packages:
        for axis in AXES:
            orientation = rotate_orientation(package.orientation, axis)
            if add(rotate_package(state, package.id, orientation)):
                return result
    return result

# Kalau mau buat deterministik bisa tambahin aj di argumen rng instance randomny
def random_neighbor(
    state: State,
    checker: ConstraintChecker | None = None,
    rng: random.Random | None = None,
    attempts: int = 40,
) -> State | None:
    checker = checker or ConstraintChecker(state.truck)
    rng = rng or random.Random()
    packages = list(state)
    if not packages:
        return None
    for _ in range(max(1, attempts)):
        move_type = rng.choice(("swap", "move", "rotate"))
        if move_type == "swap" and len(packages) >= 2:
            first, second = rng.sample(packages, 2)
            candidate = swap_packages(state, first.id, second.id)
        else:
            package = rng.choice(packages)
            if move_type == "rotate":
                choices = [rotate_orientation(package.orientation, axis) for axis in AXES]
                candidate = rotate_package(state, package.id, rng.choice(choices))
            else:
                positions = list(_positions_for(package, state))
                candidate = move_package(state, package.id, rng.choice(positions))
        if _signature(candidate) != _signature(state) and checker.is_valid(candidate):
            return candidate
    return None

def get_neighbors(state: State, checker: ConstraintChecker | None = None) -> list[State]:
    return enumerate_neighbors(state, checker)
