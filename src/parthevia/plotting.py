from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .io import load_json, state_from_dict
from .visualization import plot_state_comparison

TITLES = {
    "hill_climbing": "Hill Climbing",
    "stochastic_hill_climbing": "Stochastic Hill Climbing",
    "random_restart_hill_climbing": "Random Restart Hill Climbing",
    "simulated_annealing": "Simulated Annealing",
    "genetic_algorithm": "Genetic Algorithm",
}


def _algorithm(data: Mapping[str, Any]) -> str:
    name = data.get("algorithm", "hill_climbing" if "objectiveHistory" in data else None)
    if name not in TITLES:
        raise ValueError(f"unsupported result algorithm: {name}")
    return name


def _series(values: Any, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty numeric sequence")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
           for value in values):
        raise ValueError(f"{name} must contain finite numbers")
    return list(values)


def _segments(data: Mapping[str, Any]) -> list[tuple[list[int], list[float]]]:
    history = _series(data.get("objectiveHistory", data.get("history")), "history")
    iterations = data.get("iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if _algorithm(data) != "random_restart_hill_climbing":
        if len(history) != iterations + 1:
            raise ValueError("history must include index 0 and have iterations + 1 entries")
        return [(list(range(len(history))), history)]

    extra = data.get("extra", {})
    starts = extra.get("restart_starts", [])
    counts = extra.get("iterations_per_restart", [])
    if not starts or len(starts) != len(counts):
        raise ValueError("Random Restart requires restart_starts and iterations_per_restart")
    segments = []
    position = offset = 0
    for start, count in zip(starts, counts):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or start != position:
            raise ValueError("invalid Random Restart history boundaries")
        end = position + count + 1
        segments.append((list(range(offset, offset + count + 1)), history[position:end]))
        position = end
        offset += count
    if position != len(history) or offset != iterations:
        raise ValueError("Random Restart history does not match restart iteration counts")
    return segments


def _xlabel(algorithm: str) -> str:
    if algorithm == "genetic_algorithm":
        return "Generasi (0 = populasi awal)"
    if algorithm == "random_restart_hill_climbing":
        return "Iterasi kumulatif (awal restart berbagi indeks)"
    return "Iterasi (0 = state awal)"


def _new_plot(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def _finish(fig: Figure, ax: Axes, path: str | Path | None) -> Figure:
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150)
    return fig


def plot_objective(data: Mapping[str, Any], path: str | Path | None = None) -> Figure:
    """Plot recorded values at their original iteration or generation indices."""
    algorithm = _algorithm(data)
    segments = _segments(data)
    extra = data.get("extra", {})
    x, history = segments[0]
    secondary = None
    if algorithm == "genetic_algorithm":
        secondary = _series(extra.get("mean_history"), "mean_history")
    elif algorithm == "simulated_annealing" and "best_history" in extra:
        secondary = _series(extra["best_history"], "best_history")
    if secondary is not None and len(secondary) != len(history):
        raise ValueError("secondary history length must match history")

    fig, ax = _new_plot(TITLES[algorithm] + " - objective", _xlabel(algorithm), "Objective (total value)")
    if algorithm == "random_restart_hill_climbing":
        for index, (x, values) in enumerate(segments):
            ax.plot(x, values, marker=".", label=f"Restart {index} (0 = run awal)")
    else:
        label = "Fitness maksimum populasi" if algorithm == "genetic_algorithm" else "Objective state saat ini"
        ax.plot(x, history, color="#2563eb", marker=".", label=label)
        if secondary is not None:
            label = "Fitness rata-rata populasi" if algorithm == "genetic_algorithm" else "Objective terbaik sejauh ini"
            ax.plot(x, secondary, color="#d97706", linestyle="--", marker=".", label=label)
    return _finish(fig, ax, path)


def plot_sa_exponential(data: Mapping[str, Any], path: str | Path | None = None) -> Figure:
    """Use only logged worsening proposals; no values are filled into missing iterations."""
    if _algorithm(data) != "simulated_annealing":
        raise ValueError("SA exponential plot requires a Simulated Annealing result")
    _segments(data)
    rows = data.get("extra", {}).get("acceptance")
    if not isinstance(rows, list):
        raise ValueError("SA result is missing the recorded acceptance rows")
    previous = 0
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            raise ValueError("acceptance rows must contain [iteration, temperature, delta, exponential, accepted]")
        iteration, temperature, delta, exponential, accepted = row
        _series([temperature, delta, exponential], "acceptance values")
        if (isinstance(iteration, bool) or not isinstance(iteration, int)
                or not previous < iteration <= data["iterations"] or temperature <= 0
                or delta >= 0 or not 0 <= exponential <= 1 or not isinstance(accepted, bool)):
            raise ValueError("acceptance rows must describe ordered worsening proposals (delta < 0)")
        previous = iteration
    fig, ax = _new_plot("Simulated Annealing - eksponensial proposal memburuk",
                        "Iterasi proposal (indeks asli)", r"$e^{\Delta E/T}$ mentah ($\Delta E < 0$)")
    for accepted, label, color, marker in ((True, "Diterima", "#2563eb", "o"),
                                           (False, "Ditolak", "#d97706", "x")):
        selected = [row for row in rows if row[4] == accepted]
        if selected:
            ax.scatter([row[0] for row in selected], [row[3] for row in selected],
                       label=label, color=color, marker=marker)
    if not rows:
        ax.text(0.5, 0.5, "Tidak ada proposal memburuk yang tercatat.", transform=ax.transAxes, ha="center")
    ax.text(0.02, 0.98, "Hanya delta < 0 yang dicatat; pada sampel ini min(1, exp(delta/T)) = exp(delta/T).",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0, max(1, data["iterations"]))
    return _finish(fig, ax, path)


def plot_run_comparison(runs: Mapping[str, Mapping[str, Any]], path: str | Path | None = None) -> Figure:
    """Compare runs of one algorithm without padding, smoothing or interpolation."""
    if not runs:
        raise ValueError("at least one run is required")
    algorithms = {_algorithm(data) for data in runs.values()}
    if len(algorithms) != 1:
        raise ValueError("compare runs of the same algorithm to keep axis and history meaning consistent")
    algorithm = next(iter(algorithms))
    segments = {name: _segments(data) for name, data in runs.items()}
    suffix = "maksimum populasi" if algorithm == "genetic_algorithm" else "state saat ini"
    fig, ax = _new_plot(TITLES[algorithm] + " - perbandingan " + suffix,
                        _xlabel(algorithm), "Objective (total value)")
    for index, (name, parts) in enumerate(segments.items()):
        color = plt.get_cmap("tab10")(index % 10)
        for part, (x, values) in enumerate(parts):
            ax.plot(x, values, color=color, marker=".", label=name if part == 0 else "_nolegend_")
    return _finish(fig, ax, path)


def plot_saved_result(path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Read one saved run and export figures without executing a search algorithm."""
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("result JSON must contain one result object")
    initial = data.get("initialState", data.get("initial_state"))
    final = data.get("finalState", data.get("final_state"))
    if initial is None or final is None:
        raise ValueError("saved result must contain both initial and final states")
    initial_state, final_state = state_from_dict(initial), state_from_dict(final)
    destination = Path(output_dir)
    files = {"objective": destination / "objective.png", "states": destination / "states.png"}
    fig = plot_objective(data, files["objective"])
    plt.close(fig)
    fig = plot_state_comparison(initial_state, final_state, files["states"])
    plt.close(fig)
    if _algorithm(data) == "simulated_annealing":
        files["sa_exponential"] = destination / "sa_exponential.png"
        fig = plot_sa_exponential(data, files["sa_exponential"])
        plt.close(fig)
    return files
