from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .io import save_state, state_from_dict
from .model import ConstraintChecker, State
from .neighbors import move_package
from .visualization import plot_state_comparison


def example_states() -> tuple[State, State]:
    # Contoh tetap untuk memeriksa gambar, bukan hasil eksperimen pencarian.
    initial = state_from_dict({
        "truck": {"dimensions": [6, 4, 4], "maxCapacity": 30},
        "packages": [
            {"id": "A", "dimensions": [2, 2, 1], "value": 10, "weight": 4, "position": [0, 0, 0]},
            {"id": "B", "dimensions": [1, 2, 1], "orientation": ["y", "x", "z"],
             "value": 8, "weight": 2, "position": [0, 0, 1]},
            {"id": "C", "dimensions": [1, 1, 2], "value": 6, "weight": 3,
             "isFragile": True, "position": [3, 0, 0]},
            {"id": "D", "dimensions": [2, 1, 1], "value": 12, "weight": 4},
            {"id": "E", "dimensions": [7, 1, 1], "value": 20, "weight": 5},
        ],
    })
    final = move_package(initial, "D", (3, 2, 0))
    checker = ConstraintChecker(initial.truck)
    for state in (initial, final):
        validation = checker.check(state)
        if not validation.valid:
            raise ValueError(f"invalid example state: {validation.violations}")
    return initial, final


def main() -> None:
    initial, final = example_states()
    destination = Path("output/visualization")
    save_state(initial, destination / "initial_state.json")
    save_state(final, destination / "final_state.json")
    fig = plot_state_comparison(initial, final, destination / "state_comparison.png")
    plt.close(fig)
    print(f"Saved {destination / 'state_comparison.png'}")


if __name__ == "__main__":
    main()
