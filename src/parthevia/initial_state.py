from __future__ import annotations

import random
from typing import Iterable

from .model import ConstraintChecker, Package, State, Truck
from .neighbors import ORIENTATIONS, move_package


def drop_height(state: State, package_id: str, dims: tuple[int, int, int], x: int, y: int) -> int:
    #Ketinggian z terendah agar paket di (x, y) tidak menembus paket lain di bawahny
    height = 0
    for other in state.inside_packages:
        if other.id == package_id:
            continue
        ow, ol, oh = other.effective_dimensions
        ox, oy, oz = other.position
        if min(x + dims[0], ox + ow) > max(x, ox) and min(y + dims[1], oy + ol) > max(y, oy):
            height = max(height, oz + oh)
    return height


def random_initial_state(
    truck: Truck,
    packages: Iterable[Package],
    rng: random.Random | None = None,
    fill_prob: float = 0.7,
    attempts: int = 20,
) -> State:
    rng = rng or random.Random()
    checker = ConstraintChecker(truck)
    outside = []
    for package in packages:
        item = package.copy()
        item.position = None
        item.orientation = rng.choice(ORIENTATIONS)
        outside.append(item)
    state = State(truck, outside)

    order = list(state.packages)
    rng.shuffle(order)
    for package_id in order:
        if rng.random() >= fill_prob:
            continue
        dims = state[package_id].effective_dimensions
        if dims[0] > truck.width or dims[1] > truck.length or dims[2] > truck.height:
            continue
        for _ in range(attempts):
            x = rng.randint(0, truck.width - dims[0])
            y = rng.randint(0, truck.length - dims[1])
            z = drop_height(state, package_id, dims, x, y)
            if z + dims[2] > truck.height:
                continue
            candidate = move_package(state, package_id, (x, y, z))
            if checker.is_valid(candidate):
                state = candidate
                break
    return state
