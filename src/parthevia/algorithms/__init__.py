from .genetic_algorithm import genetic_algorithm
from .hill_climbing import (
    HillClimbing,
    HillClimbingResult,
    SidewaysHillClimbing,
    SteepestAscentHC,
    SteepestAscentHillClimbing,
)
from .random_restart_hc import random_restart_hill_climbing
from .simulated_annealing import simulated_annealing
from .stochastic_hc import stochastic_hill_climbing

# Registry of the available algorithm implementations.
ALGORITHMS = {
    "ga": genetic_algorithm,
    "sa": simulated_annealing,
    "stochastic-hc": stochastic_hill_climbing,
    "random-restart-hc": random_restart_hill_climbing,
    "steepest-ascent-hc": SteepestAscentHillClimbing,
    "sideways-hc": SidewaysHillClimbing,
}

__all__ = [
    "ALGORITHMS",
    "HillClimbing",
    "HillClimbingResult",
    "SidewaysHillClimbing",
    "SteepestAscentHC",
    "SteepestAscentHillClimbing",
    "genetic_algorithm",
    "random_restart_hill_climbing",
    "simulated_annealing",
    "stochastic_hill_climbing",
]
