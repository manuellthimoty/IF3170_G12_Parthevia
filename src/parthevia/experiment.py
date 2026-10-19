from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .algorithms import ALGORITHMS
from .initial_state import random_initial_state
from .io import load_problem, save_json
from .model import is_valid
from .plotting import plot_saved_result

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def run_experiment(name: str, problem_path: str | Path, runs: int = 3, seed: int = 0, out_dir: Path = OUTPUT_DIR, **params):
    """Jalankan algoritma `name` sebanyak `runs` kali, simpan JSON + plot ke out_dir/<name>/."""
    truck, packages = load_problem(problem_path)
    algorithm = ALGORITHMS[name]
    folder = Path(out_dir) / name
    results = []
    for run in range(1, runs + 1):
        rng = random.Random(seed + run)
        initial = random_initial_state(truck, packages, rng)
        start = time.perf_counter()
        if isinstance(algorithm, type):
            result = algorithm(**params).run(initial)
            final_objective = result.final_value
        else:
            result = algorithm(initial, rng, **params)
            result.params["seed"] = seed + run
            result.extra["duration_seconds"] = time.perf_counter() - start
            final_objective = result.final_objective
        duration = time.perf_counter() - start
        assert is_valid(result.final_state)

        save_json(result.to_dict(), folder / f"run_{run}.json")
        plot_saved_result(folder / f"run_{run}.json", folder / f"run_{run}")
        print(f"[{name}] run {run}: OF {result.history[0]:g} -> {final_objective:g}, "
              f"{result.iterations} iterasi, {duration:.2f} s")
        results.append(result)
    return results


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ALGORITHMS:
        sys.exit(f"Pemakaian: python -m parthevia.experiment <{'|'.join(ALGORITHMS)}> <input.json>")
    run_experiment(sys.argv[1], sys.argv[2])