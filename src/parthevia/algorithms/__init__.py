from .random_restart_hc import random_restart_hill_climbing
from .simulated_annealing import simulated_annealing
from .stochastic_hc import stochastic_hill_climbing

# regis algoritma buat eksprimen
ALGORITHMS = {
    "sa": simulated_annealing,
    "stochastic-hc": stochastic_hill_climbing,
    "random-restart-hc": random_restart_hill_climbing,
}

__all__ = [
    "ALGORITHMS",
    "random_restart_hill_climbing",
    "simulated_annealing",
    "stochastic_hill_climbing",
]
