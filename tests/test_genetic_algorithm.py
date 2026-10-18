from __future__ import annotations

import importlib
import random
import unittest
from unittest.mock import patch

from parthevia.algorithms import ALGORITHMS, genetic_algorithm
from parthevia.algorithms.genetic_algorithm import _crossover, _mutate
from parthevia.model import ConstraintChecker, Package, State, Truck, objective
from parthevia.result import SearchResult
from parthevia.visualization_example import example_states

ga_module = importlib.import_module("parthevia.algorithms.genetic_algorithm")


class GeneticAlgorithmTests(unittest.TestCase):
    def test_population_validity_and_history_against_evaluated_states(self) -> None:
        template, _ = example_states()
        before = template.to_dict()
        evaluated: list[tuple[State, float]] = []

        def evaluate(state: State, checker: ConstraintChecker) -> float:
            value = objective(state, checker)
            evaluated.append((state.copy(), value))
            return value

        with patch.object(ga_module, "objective", side_effect=evaluate):
            result = genetic_algorithm(template, population_size=6, max_generations=5,
                                       mutation_rate=1, seed=17)
        self.assertIsInstance(result, SearchResult)
        self.assertIs(ALGORITHMS["ga"], genetic_algorithm)
        self.assertEqual(len(evaluated), 6 * 6)
        self.assertEqual(len(result.history), 6)
        self.assertEqual(result.iterations, 5)
        checker = ConstraintChecker(template.truck)
        for state, value in evaluated:
            self.assertTrue(checker.is_valid(state))
            self.assertEqual(set(state.packages), set(template.packages))
            self.assertEqual(value, state.total_value)
            for package in state:
                original = template[package.id]
                self.assertEqual((package.dims, package.value, package.weight, package.is_fragile, package.eta),
                                 (original.dims, original.value, original.weight, original.is_fragile, original.eta))
        for generation in range(6):
            values = [value for _, value in evaluated[generation * 6:(generation + 1) * 6]]
            self.assertEqual(result.history[generation], max(values))
            self.assertAlmostEqual(result.extra["mean_history"][generation], sum(values) / len(values))
        first_generation = evaluated[:6]
        expected_initial = max(first_generation, key=lambda item: item[1])[0]
        self.assertEqual(result.initial_state.to_dict(), expected_initial.to_dict())
        self.assertEqual(result.initial_objective, objective(result.initial_state))
        self.assertEqual(result.final_objective, objective(result.final_state))
        self.assertEqual(result.final_objective, max(value for _, value in evaluated))
        self.assertEqual(result.history, sorted(result.history))
        self.assertGreaterEqual(result.extra["duration_seconds"], 0)
        self.assertEqual(template.to_dict(), before)

    def test_generation_zero_statistics_and_independent_result_states(self) -> None:
        truck = Truck([2, 1, 1], 10)
        outside = State(truck, [Package("A", [1, 1, 1], value=7)])
        inside = State(truck, [Package("A", [1, 1, 1], value=7, position=[0, 0, 0])])
        with patch.object(ga_module, "random_initial_state", side_effect=[outside, inside]):
            result = genetic_algorithm(outside, population_size=2, max_generations=0, seed=3)
        self.assertEqual(result.history, [7])
        self.assertEqual(result.extra["mean_history"], [3.5])
        self.assertTrue(result.extra["history_includes_generation_0"])
        self.assertEqual(result.initial_state.to_dict(), inside.to_dict())
        result.final_state["A"].position = None
        self.assertEqual(result.initial_state["A"].position, (0, 0, 0))
        self.assertEqual(inside["A"].position, (0, 0, 0))
        self.assertIsNone(outside["A"].position)

    def assert_same_search(self, first: SearchResult, second: SearchResult) -> None:
        self.assertEqual(first.initial_state.to_dict(), second.initial_state.to_dict())
        self.assertEqual(first.final_state.to_dict(), second.final_state.to_dict())
        self.assertEqual(first.algorithm, second.algorithm)
        self.assertEqual(first.initial_objective, second.initial_objective)
        self.assertEqual(first.final_objective, second.final_objective)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.params, second.params)
        self.assertEqual({key: value for key, value in first.extra.items() if key != "duration_seconds"},
                         {key: value for key, value in second.extra.items() if key != "duration_seconds"})

    def test_seed_and_external_rng_reproduction(self) -> None:
        template, _ = example_states()
        params = {"population_size": 5, "max_generations": 4}
        first = genetic_algorithm(template, seed=29, **params)
        self.assert_same_search(first, genetic_algorithm(template, seed=29, **params))
        rng = random.Random(31)
        rng.random()
        external = genetic_algorithm(template, rng, **params)
        replay_rng = random.Random()
        replay_rng.setstate(external.params["rng_state"])
        self.assert_same_search(external, genetic_algorithm(template, replay_rng, **params))
        generated = genetic_algorithm(template, **params)
        self.assertIsInstance(generated.params["seed"], int)
        self.assert_same_search(generated, genetic_algorithm(template, seed=generated.params["seed"], **params))

    def test_crossover_can_combine_parents_without_changing_them(self) -> None:
        truck = Truck([2, 1, 1], 10)
        first = State(truck, [Package("A", [1, 1, 1], position=[0, 0, 0]),
                              Package("B", [1, 1, 1])])
        second = State(truck, [Package("A", [1, 1, 1]),
                               Package("B", [1, 1, 1], position=[1, 0, 0])])
        before = (first.to_dict(), second.to_dict())
        children = [_crossover(first, second, ConstraintChecker(truck), random.Random(seed), 3)
                    for seed in range(8)]
        self.assertTrue(any(len(child.inside_packages) == 2 for child in children))
        self.assertTrue(all(ConstraintChecker(truck).is_valid(child) for child in children))
        children[0]["A"].position = (1, 0, 0)
        self.assertEqual((first.to_dict(), second.to_dict()), before)

    def test_invalid_crossover_has_bounded_parent_fallback(self) -> None:
        truck = Truck([1, 1, 1], 10)
        first = State(truck, [Package("A", [1, 1, 1], position=[0, 0, 0]),
                              Package("B", [1, 1, 1])])
        second = State(truck, [Package("A", [1, 1, 1]),
                               Package("B", [1, 1, 1], position=[0, 0, 0])])
        rng = random.Random(7)
        with patch.object(rng, "random", side_effect=[0.1, 0.9] * 3) as draws:
            child = _crossover(first, second, ConstraintChecker(truck), rng, 3)
        self.assertEqual(draws.call_count, 6)
        self.assertIn(child.to_dict(), (first.to_dict(), second.to_dict()))
        self.assertIsNot(child, first)
        self.assertIsNot(child, second)
        self.assertTrue(ConstraintChecker(truck).is_valid(child))

    def test_mutation_accepts_lower_fitness_and_does_not_change_parent(self) -> None:
        state = State(Truck([1, 1, 1], 10),
                      [Package("A", [1, 1, 1], value=7, position=[0, 0, 0])])
        before = state.to_dict()
        rng = random.Random(5)
        with patch.object(rng, "choice", side_effect=["move", state["A"]]), \
                patch.object(rng, "randrange", return_value=1):
            child = _mutate(state, ConstraintChecker(state.truck), rng, 1)
        self.assertLess(objective(child), objective(state))
        self.assertTrue(ConstraintChecker(state.truck).is_valid(child))
        self.assertEqual(state.to_dict(), before)

    def test_mutation_fallback_is_bounded(self) -> None:
        state = State(Truck([1, 1, 1], 0), [Package("A", [2, 2, 2])])
        rng = random.Random(3)
        with patch.object(rng, "choice", return_value="swap") as draws:
            child = _mutate(state, ConstraintChecker(state.truck), rng, 4)
        self.assertEqual(draws.call_count, 4)
        self.assertEqual(child.to_dict(), state.to_dict())
        self.assertIsNot(child["A"], state["A"])

    def test_small_and_impossible_problems_terminate(self) -> None:
        cases = [
            State(Truck([1, 1, 1], 1), []),
            State(Truck([1, 1, 1], 1), [Package("A", [1, 1, 1], value=2, weight=1)]),
            State(Truck([1, 1, 1], 1), [Package("A", [2, 2, 2], value=2)]),
            State(Truck([1, 1, 1], 0), [Package("A", [1, 1, 1], weight=1)]),
        ]
        for state in cases:
            with self.subTest(packages=len(state), capacity=state.truck.max_capacity):
                result = genetic_algorithm(state, population_size=2, max_generations=3,
                                           mutation_rate=1, crossover_attempts=1, mutation_attempts=2, seed=41)
                self.assertEqual(result.iterations, 3)
                self.assertEqual(len(result.history), 4)
                self.assertTrue(ConstraintChecker(state.truck).is_valid(result.final_state))
                self.assertEqual(set(result.final_state.packages), set(state.packages))
        self.assertEqual(result.history, [0, 0, 0, 0])

    def test_invalid_parameters(self) -> None:
        template = State(Truck([1, 1, 1], 1), [])
        invalid = [
            {"population_size": 1}, {"population_size": True}, {"population_size": 2.5},
            {"max_generations": -1}, {"max_generations": 1.5},
            {"crossover_rate": -0.1}, {"mutation_rate": 1.1},
            {"mutation_rate": float("nan")}, {"crossover_rate": True},
            {"initial_fill_prob": 2}, {"tournament_size": 0},
            {"crossover_attempts": 0}, {"mutation_attempts": 0}, {"initial_attempts": 0},
            {"seed": "7"}, {"seed": True}, {"rng": random.Random(1), "seed": 1}, {"rng": 1},
            {"rng": random.SystemRandom()},
        ]
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValueError):
                genetic_algorithm(template, **params)


if __name__ == "__main__":
    unittest.main()
