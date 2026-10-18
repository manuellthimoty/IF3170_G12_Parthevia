from __future__ import annotations

import argparse
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path
import platform
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt

from .algorithms import genetic_algorithm
from .io import load_json, load_problem_data, save_json, state_from_dict
from .model import ConstraintChecker, State
from .objectives import OBJECTIVES, get_objective
from .plotting import plot_run_comparison, plot_saved_result

SEEDS = (11, 29, 47)
CONFIGURATIONS = [
    {"id": f"A_p24_g{generations}", "series": "A", "population_size": 24, "max_generations": generations}
    for generations in (20, 60, 120)
] + [
    {"id": f"B_p{population}_g60", "series": "B", "population_size": population, "max_generations": 60}
    for population in (12, 24, 48)
]
COMMON_PARAMS = {
    "crossover_rate": 0.8, "mutation_rate": 0.2, "tournament_size": 3,
    "crossover_attempts": 10, "mutation_attempts": 40,
    "initial_fill_prob": 0.7, "initial_attempts": 20,
}


def run_ga_experiments(input_path: str | Path, output_dir: str | Path,
                       objective_name: str = "total_value") -> dict[str, Any]:
    destination = Path(output_dir)
    if any((destination / name).exists() for name in ("plan.json", "input.json", "runs")):
        raise FileExistsError("experiment output already exists; choose a new output directory")
    input_data = load_json(input_path)
    truck, packages = load_problem_data(input_data)
    score = get_objective(objective_name)
    outside = State(truck, [package.with_position(None) for package in packages])
    score(outside, ConstraintChecker(truck))
    save_json(input_data, destination / "input.json")
    input_hash = sha256((destination / "input.json").read_bytes()).hexdigest()
    plan = {
        "experiment_id": "parthevia_ga_batch4",
        "input_source": str(input_path), "input_file": "input.json", "input_sha256": input_hash,
        "dataset_source": input_data.get("source", "Source not specified by input file"),
        "objective": "total_value_of_inside_packages" if objective_name == "total_value" else objective_name,
        "objective_name": objective_name,
        "seeds": list(SEEDS), "configurations": CONFIGURATIONS, "common_params": COMMON_PARAMS,
        "expected_runs": len(CONFIGURATIONS) * len(SEEDS),
        "history_includes_generation_0": True,
        "selection_rationale": (
            "Pilot terpisah p12/g10 dan p24/g20 selesai sekitar 0.10 dan 0.27 detik. "
            "Rentang generasi 20/60/120 dan populasi 12/24/48 memperluas anggaran pencarian "
            "dengan waktu wajar; perubahan hasil tidak dijamin. Operator tetap default GA. "
            "A_p24_g60 dan B_p24_g60 sengaja diulang terpisah sebagai kontrol kedua rangkaian."
        ),
        "python_version": platform.python_version(),
        "algorithm_sha256": sha256((Path(__file__).parent / "algorithms/genetic_algorithm.py").read_bytes()).hexdigest(),
    }
    save_json(plan, destination / "plan.json")
    template = State(truck, packages)
    for config in CONFIGURATIONS:
        for repeat, seed in enumerate(SEEDS, 1):
            run_id = f"{config['id']}_seed{seed}"
            result = genetic_algorithm(template, population_size=config["population_size"],
                                       max_generations=config["max_generations"], seed=seed,
                                       objective_name=objective_name, **COMMON_PARAMS)
            data = result.to_dict()
            data["experiment"] = {
                "run_id": run_id, "configuration_id": config["id"], "series": config["series"],
                "repeat": repeat, "seed": seed, "input_file": "../input.json", "input_sha256": input_hash,
                "objective": plan["objective"], "requested_generations": config["max_generations"],
                "actual_generations": result.iterations,
                "completed_as_planned": result.iterations == config["max_generations"],
            }
            save_json(data, destination / "runs" / f"{run_id}.json")
            print(f"{run_id}: objective={result.final_objective:g}, generations={result.iterations}, "
                  f"duration={result.extra['duration_seconds']:.3f}s", flush=True)
    return summarize_ga_experiments(destination)


def verify_ga_experiments(output_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(output_dir)
    plan = load_json(root / "plan.json")
    objective_name = plan.get("objective_name", "total_value")
    score = get_objective(objective_name)
    input_hash = sha256((root / plan["input_file"]).read_bytes()).hexdigest()
    if input_hash != plan["input_sha256"]:
        raise ValueError("input snapshot hash does not match plan")
    truck, packages = load_problem_data(load_json(root / plan["input_file"]))
    template = State(truck, packages)
    checker = ConstraintChecker(truck)
    expected = {f"{config['id']}_seed{seed}" for config in plan["configurations"] for seed in plan["seeds"]}
    files = list((root / "runs").glob("*.json"))
    if len(expected) != plan["expected_runs"] or {path.stem for path in files} != expected:
        raise ValueError("run files do not match the complete planned matrix")
    records = []
    for config in plan["configurations"]:
        for repeat, seed in enumerate(plan["seeds"], 1):
            run_id = f"{config['id']}_seed{seed}"
            data = load_json(root / "runs" / f"{run_id}.json")
            metadata = data["experiment"]
            expected_metadata = {
                "run_id": run_id, "configuration_id": config["id"], "series": config["series"],
                "repeat": repeat, "seed": seed, "input_file": "../input.json", "input_sha256": input_hash,
                "objective": plan["objective"], "requested_generations": config["max_generations"],
            }
            expected_params = dict(plan["common_params"], population_size=config["population_size"],
                                   max_generations=config["max_generations"], seed=seed, elite_count=1)
            if "objective_name" in plan:
                expected_params["objective_name"] = objective_name
            if (data["algorithm"] != "genetic_algorithm"
                    or any(metadata.get(key) != value for key, value in expected_metadata.items())
                    or any(data["params"].get(key) != value for key, value in expected_params.items())):
                raise ValueError(f"{run_id}: configuration, seed or input metadata mismatch")
            actual = data["iterations"]
            if (not isinstance(actual, int) or isinstance(actual, bool) or not 0 <= actual <= config["max_generations"]
                    or metadata["actual_generations"] != actual
                    or metadata["completed_as_planned"] != (actual == config["max_generations"])):
                raise ValueError(f"{run_id}: incorrect actual generation count")
            for key, value_key in (("initial_state", "initial_objective"), ("final_state", "final_objective")):
                state = state_from_dict(data[key])
                if state.truck != truck or set(state.packages) != set(template.packages) or not checker.is_valid(state):
                    raise ValueError(f"{run_id}: invalid {key} or missing package IDs")
                for package in state:
                    original = template[package.id]
                    if (package.dims, package.value, package.weight, package.is_fragile, package.eta) != (
                            original.dims, original.value, original.weight, original.is_fragile, original.eta):
                        raise ValueError(f"{run_id}: package attributes changed")
                if not isclose(score(state, checker), data[value_key]):
                    raise ValueError(f"{run_id}: objective does not match {key}")
                total_key = "initial_total_value" if key == "initial_state" else "final_total_value"
                if (total_key in data["extra"] and not isclose(state.total_value, data["extra"][total_key])
                        or objective_name != "total_value" and total_key not in data["extra"]):
                    raise ValueError(f"{run_id}: missing or incorrect original total value")
            maximum, average = data["history"], data["extra"]["mean_history"]
            duration = data["extra"]["duration_seconds"]
            if (len(maximum) != actual + 1 or len(average) != len(maximum)
                    or not data["extra"].get("history_includes_generation_0")
                    or not all(isfinite(value) for value in maximum + average)
                    or any(a > b for a, b in zip(maximum, maximum[1:]))
                    or any(avg > best + 1e-9 for avg, best in zip(average, maximum))
                    or not isclose(maximum[0], data["initial_objective"])
                    or not isclose(maximum[-1], data["final_objective"])
                    or not isfinite(duration) or duration < 0):
                raise ValueError(f"{run_id}: inconsistent history or duration")
            records.append(data)
    return records


def summarize_ga_experiments(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    records = verify_ga_experiments(root)
    rows = []
    for config in load_json(root / "plan.json")["configurations"]:
        runs = [data for data in records if data["experiment"]["configuration_id"] == config["id"]]
        objectives = [data["final_objective"] for data in runs]
        durations = [data["extra"]["duration_seconds"] for data in runs]
        totals = [state_from_dict(data["final_state"]).total_value for data in runs]
        rows.append(dict(config, run_ids=[data["experiment"]["run_id"] for data in runs],
                         seeds=[data["params"]["seed"] for data in runs],
                         final_objectives=objectives, objective_mean=mean(objectives),
                         objective_sample_sd=stdev(objectives), duration_mean_seconds=mean(durations),
                         duration_sample_sd_seconds=stdev(durations), final_total_values=totals,
                         total_value_mean=mean(totals)))
    summary = {
        "run_count": len(records),
        "objective_name": load_json(root / "plan.json").get("objective_name", "total_value"),
        "completed_as_planned": sum(data["experiment"]["completed_as_planned"] for data in records),
        "variation_measure": "sample standard deviation, denominator n - 1, n = 3 per configuration",
        "verification": "All saved initial/final states, IDs, objectives, seeds, parameters and history lengths checked.",
        "configurations": rows,
    }
    save_json(summary, root / "summary.json")
    return summary


def plot_ga_experiments(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    records = verify_ga_experiments(root)
    files = []
    for config in load_json(root / "plan.json")["configurations"]:
        runs = {f"Seed {data['params']['seed']}": data for data in records
                if data["experiment"]["configuration_id"] == config["id"]}
        fig = plot_run_comparison(runs)
        fig.axes[0].set_title(f"GA {config['series']} - populasi {config['population_size']}, "
                              f"generasi {config['max_generations']}")
        path = root / "plots" / f"{config['id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files.append(str(path.relative_to(root)))
    selected = max(records, key=lambda data: data["final_objective"])
    run_id = selected["experiment"]["run_id"]
    exported = plot_saved_result(root / "runs" / f"{run_id}.json", root / "plots/representative_best")
    files.extend(str(path.relative_to(root)) for path in exported.values())
    info = {"representative_run_id": run_id,
            "selection_rule": "Highest final objective across 18 runs; ties use first run in plan order.",
            "files": files}
    save_json(info, root / "plots.json")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksperimen khusus GA anggota C, bukan runner umum.")
    parser.add_argument("--input", default="data/ga_synthetic.json")
    parser.add_argument("--output", default="output/ga_experiments_batch4")
    parser.add_argument("--mode", choices=("run", "verify", "plot"), default="run")
    parser.add_argument("--objective", choices=tuple(OBJECTIVES), default="total_value")
    args = parser.parse_args()
    if args.mode == "run":
        summary = run_ga_experiments(args.input, args.output, objective_name=args.objective)
        plot_ga_experiments(args.output)
        print(f"Verified {summary['run_count']} runs; {summary['completed_as_planned']} completed as planned.")
    elif args.mode == "verify":
        summary = summarize_ga_experiments(args.output)
        print(f"Verified {summary['run_count']} runs; {summary['completed_as_planned']} completed as planned.")
    else:
        print(plot_ga_experiments(args.output))


if __name__ == "__main__":
    main()
