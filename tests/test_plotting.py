from __future__ import annotations

from copy import deepcopy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from parthevia.algorithms import (
    SidewaysHillClimbing,
    SteepestAscentHillClimbing,
    genetic_algorithm,
    random_restart_hill_climbing,
    simulated_annealing,
    stochastic_hill_climbing,
)
from parthevia.io import load_json, save_json
from parthevia.model import Package, State, Truck
from parthevia.plotting import plot_objective, plot_run_comparison, plot_sa_exponential, plot_saved_result


class PlottingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        state = State(Truck([3, 2, 2], 10), [
            Package("A", [1, 1, 1], value=5, weight=1, position=[0, 0, 0]),
            Package("B", [2, 1, 1], value=9, weight=1),
            Package("C", [1, 1, 1], value=4, weight=1),
        ])
        # Run kecil nyata; seluruh plotting sesudah ini hanya membaca hasil.
        cls.results = {
            "hc": SteepestAscentHillClimbing(max_iterations=3).run(state),
            "sideways": SidewaysHillClimbing(max_iterations=4).run(state),
            "stochastic": stochastic_hill_climbing(state, random.Random(7), max_iterations=3, neighborhood="full"),
            "restart": random_restart_hill_climbing(state, random.Random(7), max_restart=2,
                                                    inner_params={"max_iterations": 3, "neighborhood": "full"}),
            "sa": simulated_annealing(state, random.Random(7), initial_temperature=5, max_iterations=25),
            "ga": genetic_algorithm(state, population_size=6, max_generations=4, seed=7),
        }

    def tearDown(self) -> None:
        plt.close("all")

    def test_saved_real_runs_render_states_and_histories_without_search_or_mutation(self) -> None:
        before = {name: deepcopy(result.to_dict()) for name, result in self.results.items()}
        with tempfile.TemporaryDirectory() as directory:
            for name, result in self.results.items():
                with self.subTest(algorithm=name):
                    source = Path(directory) / f"{name}.json"
                    save_json(result.to_dict(), source)
                    source_bytes = source.read_bytes()
                    # Penghambat pada pintu masuk pencarian dan operator yang diperlukan tiap run.
                    with patch("parthevia.algorithms.hill_climbing.SteepestAscentHillClimbing.run",
                               side_effect=AssertionError("plot must not run HC")), \
                            patch("parthevia.algorithms.genetic_algorithm.random_initial_state",
                                  side_effect=AssertionError("plot must not initialize a population")), \
                            patch("parthevia.algorithms.simulated_annealing.random_neighbor",
                                  side_effect=AssertionError("plot must not search neighbors")), \
                            patch("parthevia.algorithms.stochastic_hc._uphill_neighbors",
                                  side_effect=AssertionError("plot must not run stochastic/restart search")):
                        files = plot_saved_result(source, Path(directory) / name)
                    self.assertEqual(set(files), {"objective", "states", "sa_exponential"} if name == "sa"
                                     else {"objective", "states"})
                    for path in files.values():
                        self.assertGreater(path.stat().st_size, 1000)
                        image = plt.imread(path)
                        self.assertGreaterEqual(image.shape[1], 1000)
                        self.assertGreaterEqual(image.shape[0], 700)
                    self.assertEqual(source.read_bytes(), source_bytes)
                    self.assertEqual(load_json(source)["iterations"], result.iterations)
        self.assertEqual({name: result.to_dict() for name, result in self.results.items()}, before)
        self.assertEqual(plt.get_fignums(), [])
        self.assertTrue(callable(self.results["restart"].params["inner"]))
        self.assertEqual(before["restart"]["params"]["inner"], "stochastic_hill_climbing")

    def test_real_hc_sa_ga_axes_values_and_legends(self) -> None:
        for name in ("hc", "sideways", "stochastic", "sa", "ga"):
            with self.subTest(algorithm=name):
                result = self.results[name]
                fig = plot_objective(result.to_dict())
                ax = fig.axes[0]
                self.assertEqual(list(ax.lines[0].get_xdata()), list(range(len(result.history))))
                self.assertEqual(list(ax.lines[0].get_ydata()), result.history)
                labels = [text.get_text() for text in ax.get_legend().get_texts()]
                self.assertEqual(len(labels), 2 if name in ("sa", "ga") else 1)
                if name == "ga":
                    self.assertIn("Generasi", ax.get_xlabel())
                    self.assertEqual(list(ax.lines[1].get_ydata()), result.extra["mean_history"])
                elif name == "sa":
                    self.assertIn("saat ini", labels[0])
                    self.assertIn("sejauh ini", labels[1])
                    self.assertEqual(list(ax.lines[1].get_ydata()), result.extra["best_history"])

    def test_real_restart_boundaries_use_cumulative_steps(self) -> None:
        result = self.results["restart"]
        ax = plot_objective(result.to_dict()).axes[0]
        offset = 0
        for line, start, count in zip(ax.lines, result.extra["restart_starts"],
                                      result.extra["iterations_per_restart"]):
            self.assertEqual(list(line.get_xdata()), list(range(offset, offset + count + 1)))
            self.assertEqual(list(line.get_ydata()), result.history[start:start + count + 1])
            offset += count
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual(offset, result.iterations)

    def test_real_sa_exponentials_use_recorded_indices_and_values(self) -> None:
        result = self.results["sa"]
        self.assertTrue(result.extra["acceptance"])
        ax = plot_sa_exponential(result.to_dict()).axes[0]
        actual = sorted(tuple(point) for collection in ax.collections for point in collection.get_offsets())
        expected = sorted((row[0], row[3]) for row in result.extra["acceptance"])
        self.assertEqual(actual, expected)
        self.assertTrue(all(row[2] < 0 for row in result.extra["acceptance"]))
        self.assertIn("mentah", ax.get_ylabel())
        self.assertTrue(any("min(1" in text.get_text() for text in ax.texts))

    def test_comparison_keeps_unequal_run_lengths(self) -> None:
        # Fixture sintetis hanya untuk edge case panjang histori dan indeks 0.
        runs = {
            "Uji panjang": {"objectiveHistory": [2, 3, 7], "iterations": 2},
            "Uji pendek": {"objectiveHistory": [4], "iterations": 0},
        }
        before = deepcopy(runs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            ax = plot_run_comparison(runs, path).axes[0]
            self.assertTrue(path.is_file())
            self.assertEqual([list(line.get_xdata()) for line in ax.lines], [[0, 1, 2], [0]])
            self.assertEqual([list(line.get_ydata()) for line in ax.lines], [[2, 3, 7], [4]])
            self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], list(runs))
        self.assertEqual(runs, before)
        with self.assertRaises(ValueError):
            plot_run_comparison({"HC": runs["Uji pendek"], "GA": self.results["ga"].to_dict()})

    def test_empty_acceptance_and_generation_zero(self) -> None:
        # Fixture sintetis: SA berhenti sebelum proposal pertama.
        data = {"algorithm": "simulated_annealing", "history": [5], "iterations": 0,
                "extra": {"acceptance": []}}
        fig = plot_sa_exponential(data)
        fig.canvas.draw()
        self.assertFalse(fig.axes[0].collections)
        self.assertTrue(any("Tidak ada" in text.get_text() for text in fig.axes[0].texts))
        data = {"algorithm": "genetic_algorithm", "history": [5], "iterations": 0,
                "extra": {"mean_history": [3]}}
        ax = plot_objective(data).axes[0]
        self.assertEqual([list(line.get_xdata()) for line in ax.lines], [[0], [0]])

    def test_sa_rejected_proposals_remain_visible_at_original_indices(self) -> None:
        # Fixture sintetis untuk proposal ditolak dan gap pada log acceptance.
        data = {"algorithm": "simulated_annealing", "history": [5, 4, 4, 4], "iterations": 3,
                "extra": {"acceptance": [[1, 2, -1, 0.6065, True], [3, 1, -2, 0.1353, False]]}}
        before = deepcopy(data)
        ax = plot_sa_exponential(data).axes[0]
        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], ["Diterima", "Ditolak"])
        self.assertEqual([list(collection.get_offsets()[:, 0]) for collection in ax.collections], [[1], [3]])
        self.assertFalse(ax.lines)
        self.assertEqual(data, before)

    def test_incomplete_data_is_rejected_without_inventing_values(self) -> None:
        data = deepcopy(self.results["ga"].to_dict())
        data["extra"]["mean_history"].pop()
        with self.assertRaises(ValueError):
            plot_objective(data)
        data = deepcopy(self.results["sa"].to_dict())
        del data["extra"]["acceptance"]
        with self.assertRaises(ValueError):
            plot_sa_exponential(data)
        data = deepcopy(self.results["restart"].to_dict())
        data["extra"]["restart_starts"][1] += 1
        with self.assertRaises(ValueError):
            plot_objective(data)
        with self.assertRaises(ValueError):
            plot_objective({"objectiveHistory": [], "iterations": 0})


if __name__ == "__main__":
    unittest.main()
