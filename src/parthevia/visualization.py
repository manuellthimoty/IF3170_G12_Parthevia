from __future__ import annotations

import itertools
from pathlib import Path
from textwrap import wrap
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

from .model import State

Color = tuple[float, float, float, float]


def package_colors(*states: State) -> dict[str, Color]:
    ids = sorted({package.id for state in states for package in state})
    palette = plt.get_cmap("tab20")
    return {package_id: palette(index % 20) for index, package_id in enumerate(ids)}


def _draw_state(ax: Axes3D, state: State, colors: Mapping[str, Color], limits: Sequence[int]) -> None:
    # Gambar hanya 12 rusuk peti kemas, tanpa enumerasi sel di dalamnya.
    corners = list(itertools.product(*((0, size) for size in state.truck.effective_dimensions)))
    for first, second in itertools.combinations(corners, 2):
        if sum(a != b for a, b in zip(first, second)) == 1:
            ax.plot(*zip(first, second), color="#334155", linewidth=1)

    for package in state.inside_packages:
        x, y, z = package.position
        width, length, height = package.effective_dimensions
        ax.bar3d(x, y, z, width, length, height, color=colors[package.id],
                 edgecolor="#334155", linewidth=0.7, alpha=0.65, shade=True)
        ax.text(x + width / 2, y + length / 2, z + height / 2, package.id,
                ha="center", va="center", fontsize=9, zorder=len(state.packages) + 10,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 1})

    ax.set(xlim=(0, limits[0]), ylim=(0, limits[1]), zlim=(0, limits[2]),
           xlabel="x (width)", ylabel="y (length)", zlabel="z (height)")
    ax.set_box_aspect(limits)
    ax.view_init(elev=24, azim=-55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=6, integer=True))


def _package_lines(state: State) -> list[tuple[str, str | None]]:
    lines: list[tuple[str, str | None]] = [("Di dalam: ID | posisi | dimensi efektif (x, y, z)", None)]
    for package in state.inside_packages:
        text = f"{package.id} | {package.position} | {package.effective_dimensions}"
        lines.extend((line, package.id) for line in wrap(text, width=65))
    if not state.inside_packages:
        lines.append(("(tidak ada)", None))
    lines.append((f"Di luar ({len(state.outside_packages)}): tanpa koordinat", None))
    for package in state.outside_packages:
        text = f"{package.id} | dimensi {package.effective_dimensions} | value {package.value:g}"
        lines.extend((line, package.id) for line in wrap(text, width=65))
    if not state.outside_packages:
        lines.append(("(tidak ada)", None))
    return lines


def _plot_states(states: Sequence[State], titles: Sequence[str], path: str | Path | None,
                 colors: Mapping[str, Color] | None) -> Figure:
    colors = colors if colors is not None else package_colors(*states)
    missing = {package.id for state in states for package in state} - colors.keys()
    if missing:
        raise ValueError(f"missing package colors: {sorted(missing)}")
    limits = tuple(max(state.truck.effective_dimensions[index] for state in states) for index in range(3))
    lines = [_package_lines(state) for state in states]
    text_height = max(len(items) for items in lines) * 0.25 + 0.3
    fig = plt.figure(figsize=(7 * len(states), 5.5 + text_height), layout="constrained")
    grid = fig.add_gridspec(2, len(states), height_ratios=(5.5, text_height))
    for index, (state, title) in enumerate(zip(states, titles)):
        ax = fig.add_subplot(grid[0, index], projection="3d")
        _draw_state(ax, state, colors, limits)
        ax.set_title(f"{title}\nTotal value: {state.total_value:g} | Berat: {state.total_weight:g}/{state.truck.max_capacity:g}")
        info = fig.add_subplot(grid[1, index])
        info.set_axis_off()
        for row, (line, package_id) in enumerate(lines[index]):
            y = 1 - (row + 0.5) / max(len(items) for items in lines)
            if package_id is not None:
                info.plot(0.02, y, marker="s", color=colors[package_id],
                          markeredgecolor="#334155", transform=info.transAxes)
            info.text(0.05, y, line, transform=info.transAxes, va="center", fontsize=10)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150)
    return fig


def plot_state(state: State, path: str | Path | None = None, title: str = "State",
               colors: Mapping[str, Color] | None = None) -> Figure:
    """Return a figure without changing state; the caller closes the figure."""
    return _plot_states((state,), (title,), path, colors)


def plot_state_comparison(initial_state: State, final_state: State,
                          path: str | Path | None = None,
                          colors: Mapping[str, Color] | None = None) -> Figure:
    """Compare states with shared axis limits, box aspect, view and package colors."""
    return _plot_states((initial_state, final_state), ("Initial state", "Final state"), path, colors)
