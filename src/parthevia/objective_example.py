from __future__ import annotations

import argparse
from pathlib import Path

from .algorithms import genetic_algorithm
from .io import load_json, load_problem_data, save_json
from .model import ConstraintChecker, State
from .objectives import OBJECTIVES, eta_priority_objective
from .plotting import plot_saved_result


def run_objective_example(input_path: str | Path, output_dir: str | Path, seed: int = 17) -> dict:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("demo output already exists; choose a new output directory")
    input_data = load_json(input_path)
    truck, packages = load_problem_data(input_data)
    template = State(truck, packages)
    checker = ConstraintChecker(truck)
    save_json(input_data, destination / "input.json")
    rows = []
    for name in OBJECTIVES:
        result = genetic_algorithm(template, population_size=12, max_generations=12,
                                   seed=seed, objective_name=name)
        if not checker.is_valid(result.initial_state) or not checker.is_valid(result.final_state):
            raise ValueError("demonstration produced an invalid state")
        data = result.to_dict()
        data["demonstration"] = {"input_file": "input.json", "input_source": str(input_path),
                                 "purpose": "Small real run showing the objective trade-off; not a full experiment."}
        path = destination / f"{name}.json"
        save_json(data, path)
        plot_saved_result(path, destination / name)
        rows.append({"objective_name": name, "objective_score": result.final_objective,
                     "total_value": result.final_state.total_value,
                     "eta_priority_score": eta_priority_objective(result.final_state, checker),
                     "inside_ids": [package.id for package in result.final_state.inside_packages]})
    summary = {"seed": seed, "population_size": 12, "max_generations": 12,
               "dataset_source": input_data.get("source"), "runs": rows}
    save_json(summary, destination / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Dua run GA pada input dan seed sama, objective berbeda.")
    parser.add_argument("--input", default="data/eta_demo.json")
    parser.add_argument("--output", default="output/eta_objective_demo")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(run_objective_example(args.input, args.output, args.seed))


if __name__ == "__main__":
    main()
