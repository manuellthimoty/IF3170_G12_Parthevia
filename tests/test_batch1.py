from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from parthevia.algorithms import (
    ALGORITHMS,
    SidewaysHillClimbing,
    SteepestAscentHillClimbing,
    random_restart_hill_climbing,
    simulated_annealing,
    stochastic_hill_climbing,
)
from parthevia.io import load_state, save_state
from parthevia.model import ConstraintChecker, Package, State, Truck, objective
from parthevia.result import SearchResult
from parthevia.visualization import package_colors, plot_state, plot_state_comparison
from parthevia.visualization_example import example_states


class Batch1Tests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_algorithm_imports_and_result_contract(self) -> None:
        self.assertTrue({"sa", "stochastic-hc", "random-restart-hc",
                         "steepest-ascent-hc", "sideways-hc"}.issubset(ALGORITHMS))
        state = State(Truck([2, 2, 2], 10), [Package("P", [1, 1, 1], value=3, weight=1)])
        before = state.to_dict()
        results = [
            SteepestAscentHillClimbing(max_iterations=2).run(state),
            SidewaysHillClimbing(max_iterations=2).run(state),
            stochastic_hill_climbing(state, random.Random(7), max_iterations=2, neighborhood="full"),
            simulated_annealing(state, random.Random(7), initial_temperature=10, max_iterations=2),
            random_restart_hill_climbing(state, random.Random(7), max_restart=1,
                                         inner_params={"max_iterations": 2, "neighborhood": "full"}),
        ]
        for result in results:
            self.assertTrue(ConstraintChecker(state.truck).is_valid(result.final_state))
            self.assertTrue(result.history)
            if isinstance(result, SearchResult):
                self.assertEqual(result.final_objective, objective(result.final_state))
        self.assertEqual(state.to_dict(), before)

    def test_render_comparison_and_io_without_mutation(self) -> None:
        initial, final = example_states()
        before = (initial.to_dict(), final.to_dict())
        self.assertEqual(initial["B"].effective_dimensions, (2, 1, 1))
        self.assertEqual((objective(initial), objective(final)), (24, 36))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/comparison.png"
            fig = plot_state_comparison(initial, final, path)
            self.assertGreater(path.stat().st_size, 1000)
            self.assertGreater(plt.imread(path).shape[1], 1000)
            save_state(initial, Path(directory) / "state.json")
            self.assertEqual(load_state(Path(directory) / "state.json").to_dict(), before[0])
        left, left_info, right, right_info = fig.axes
        self.assertEqual(left.get_xlim(), right.get_xlim())
        self.assertEqual(left.get_ylim(), right.get_ylim())
        self.assertEqual(left.get_zlim(), right.get_zlim())
        self.assertEqual(tuple(left.get_box_aspect()), tuple(right.get_box_aspect()))
        self.assertEqual((left.elev, left.azim), (right.elev, right.azim))
        self.assertTrue(any("Di luar (2)" in text.get_text() for text in left_info.texts))
        self.assertTrue(any("D | dimensi" in text.get_text() for text in left_info.texts))
        self.assertTrue(any("E | dimensi" in text.get_text() for text in right_info.texts))
        self.assertEqual((initial.to_dict(), final.to_dict()), before)

    def test_geometry_and_colors_follow_ids_not_order(self) -> None:
        initial, final = example_states()
        final = State(final.truck, list(reversed(list(final))))
        colors = package_colors(initial, final)
        with patch.object(Axes3D, "bar3d", autospec=True) as draw:
            plot_state_comparison(initial, final)
        calls = draw.call_args_list
        expected = initial.inside_packages + final.inside_packages
        self.assertEqual(len(calls), len(expected))
        for call, package in zip(calls, expected):
            self.assertEqual(call.args[1:4], package.position)
            self.assertEqual(call.args[4:7], package.effective_dimensions)
            self.assertEqual(call.kwargs["color"], colors[package.id])

    def test_empty_and_all_outside_states(self) -> None:
        empty = State(Truck([3, 4, 2], 10), [])
        outside = State(empty.truck, [Package("OUT", [1, 1, 1])])
        fig = plot_state_comparison(empty, outside)
        fig.canvas.draw()
        self.assertTrue(any("OUT | dimensi" in text.get_text() for text in fig.axes[3].texts))
        self.assertEqual(len(fig.axes[2].collections), 0)
        fig = plot_state(empty)
        fig.canvas.draw()
        self.assertTrue(any("Di luar (0)" in text.get_text() for text in fig.axes[1].texts))


if __name__ == "__main__":
    unittest.main()
