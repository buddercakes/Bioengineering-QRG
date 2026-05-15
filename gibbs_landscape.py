"""Thermodynamic cognitive landscape simulation utilities.

This module converts the exploratory ``Gibbslandscape.ipynb`` notebook into a
small, importable Python API. It models cognition as a toy Gibbs free-energy
landscape coupled to a simplified metabolic regulator and circadian rhythm.

Important
---------
The equations here are a conceptual simulation scaffold, not a validated
clinical or physiological model. Use the outputs for hypothesis generation,
teaching, or prototyping rather than diagnosis or treatment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


__all__ = [
    "DEFAULT_NETWORKS",
    "CircadianProfile",
    "MetabolicRegulator",
    "ThermodynamicCognitiveState",
    "SimulationResult",
    "get_circadian_factor",
    "run_circadian_simulation",
    "run_sleep_duration_sweep",
    "sample_gibbs_surface",
]


DEFAULT_NETWORKS = (
    "Executive Control",
    "Salience",
    "Default Mode",
    "Visual",
    "Auditory",
    "Sensorimotor",
    "Limbic",
    "Attention",
)


@dataclass(frozen=True)
class CircadianProfile:
    """Hourly modulation factors used by the metabolic regulator.

    Parameters
    ----------
    hourly_factors:
        Twenty-four positive factors indexed by hour of day. Values above one
        increase active metabolic flux; values below one reduce it.
    sleep_factor:
        Override factor used during the configured sleep interval.
    """

    hourly_factors: tuple[float, ...] = (
        0.95,
        0.93,
        0.92,
        0.91,
        0.92,
        0.95,
        1.00,
        1.05,
        1.08,
        1.10,
        1.08,
        1.05,
        1.02,
        1.00,
        1.01,
        1.04,
        1.07,
        1.08,
        1.05,
        1.00,
        0.97,
        0.95,
        0.94,
        0.94,
    )
    sleep_factor: float = 0.8

    def __post_init__(self) -> None:
        if len(self.hourly_factors) != 24:
            raise ValueError("hourly_factors must contain exactly 24 values.")
        if any(factor <= 0 for factor in self.hourly_factors):
            raise ValueError("all hourly circadian factors must be positive.")
        if self.sleep_factor <= 0:
            raise ValueError("sleep_factor must be positive.")


def _is_sleeping(hour_of_day: int, sleep_start_hour: int, sleep_end_hour: int) -> bool:
    """Return whether ``hour_of_day`` falls in the sleep interval."""

    hour = int(hour_of_day) % 24
    start = int(sleep_start_hour) % 24
    end = int(sleep_end_hour) % 24
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def get_circadian_factor(
    hour_of_day: int,
    sleep_start_hour: int = 22,
    sleep_end_hour: int = 6,
    profile: CircadianProfile | None = None,
) -> float:
    """Return the metabolic modulation factor for an hour of the day.

    Sleep intervals that cross midnight, such as 22:00 to 06:00, are handled
    explicitly. The function is defined before the simulation code so scripts
    can call it without notebook-order dependencies.
    """

    profile = profile or CircadianProfile()
    hour = int(hour_of_day) % 24
    if _is_sleeping(hour, sleep_start_hour, sleep_end_hour):
        return profile.sleep_factor
    return profile.hourly_factors[hour]


@dataclass
class MetabolicRegulator:
    """Track fuel, waste, and arousal terms for the landscape model.

    ``glucose`` is treated as a slow energy reservoir, ``atp`` as immediate
    metabolic currency, ``entropy`` as accumulated waste/noise, and
    ``temperature`` as an arousal-like term that scales state transitions.
    """

    glucose: float = 1000.0
    atp: float = 100.0
    entropy: float = 0.0
    temperature: float = 1.0
    clearance_rate: float = 0.1
    max_atp: float = 100.0
    min_atp: float = 0.1
    min_temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.glucose < 0:
            raise ValueError("glucose must be non-negative.")
        if self.atp <= 0:
            raise ValueError("atp must be positive.")
        if self.entropy < 0:
            raise ValueError("entropy must be non-negative.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.clearance_rate < 0:
            raise ValueError("clearance_rate must be non-negative.")

    @property
    def ATP(self) -> float:  # pragma: no cover - compatibility alias
        """Compatibility alias for notebook-style code."""

        return self.atp

    @ATP.setter
    def ATP(self, value: float) -> None:  # pragma: no cover - compatibility alias
        self.atp = value

    @property
    def Glucose(self) -> float:  # pragma: no cover - compatibility alias
        """Compatibility alias for notebook-style code."""

        return self.glucose

    @Glucose.setter
    def Glucose(self, value: float) -> None:  # pragma: no cover - compatibility alias
        self.glucose = value

    @property
    def S(self) -> float:  # pragma: no cover - compatibility alias
        """Compatibility alias for notebook-style code."""

        return self.entropy

    @S.setter
    def S(self, value: float) -> None:  # pragma: no cover - compatibility alias
        self.entropy = value

    @property
    def T(self) -> float:  # pragma: no cover - compatibility alias
        """Compatibility alias for notebook-style code."""

        return self.temperature

    @T.setter
    def T(self, value: float) -> None:  # pragma: no cover - compatibility alias
        self.temperature = value

    def perform_metabolic_cycle(
        self,
        x_activity: Iterable[float],
        current_hour: int,
        sleep_start_hour: int = 22,
        sleep_end_hour: int = 6,
        dt: float = 0.1,
        profile: CircadianProfile | None = None,
    ) -> None:
        """Advance fuel, entropy, and temperature by one simulation step."""

        if dt <= 0:
            raise ValueError("dt must be positive.")

        work_done = sum(abs(value) for value in x_activity)
        circadian_factor = get_circadian_factor(
            current_hour,
            sleep_start_hour=sleep_start_hour,
            sleep_end_hour=sleep_end_hour,
            profile=profile,
        )

        if _is_sleeping(current_hour, sleep_start_hour, sleep_end_hour):
            self.entropy -= self.entropy * self.clearance_rate * dt * 2.0
            self.atp += (self.max_atp - self.atp) * 0.5 * dt
            self.temperature = self.min_temperature
        else:
            self.atp -= work_done * 0.05 * dt * circadian_factor
            self.entropy += work_done * 0.1 * dt

            if self.glucose > 0 and self.atp < self.max_atp:
                glucose_burn = min(self.glucose, 0.01 * dt * circadian_factor)
                self.glucose -= glucose_burn
                self.atp += glucose_burn * 30.0

            self.temperature = 1.0 + self.entropy * 0.2

        self.atp = min(self.max_atp, max(self.min_atp, self.atp))
        self.entropy = max(0.0, self.entropy)
        self.glucose = max(0.0, self.glucose)
        self.temperature = max(self.min_temperature, self.temperature)


class ThermodynamicCognitiveState:
    """Navigate a toy Gibbs free-energy landscape, ``G = H - T*S``."""

    def __init__(
        self,
        n_networks: int = 8,
        networks: Iterable[str] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if n_networks <= 0:
            raise ValueError("n_networks must be positive.")

        self.rng = rng or random.Random()
        default_networks = tuple(networks or DEFAULT_NETWORKS)
        if len(default_networks) < n_networks:
            extra = tuple(f"Network {idx + 1}" for idx in range(len(default_networks), n_networks))
            default_networks = default_networks + extra
        self.networks = list(default_networks[:n_networks])
        self.n = n_networks
        self.X = [self.rng.uniform(-1.0, 1.0) for _ in range(self.n)]

        raw_weights = [
            [self.rng.gauss(0.0, 0.5) for _ in range(self.n)]
            for _ in range(self.n)
        ]
        self.H_weights = [
            [0.0 if row == column else (raw_weights[row][column] + raw_weights[column][row]) / 2.0
             for column in range(self.n)]
            for row in range(self.n)
        ]

    def update_state(
        self,
        regulator: MetabolicRegulator,
        external_drive: float | Sequence[float] = 0.0,
    ) -> list[float]:
        """Update the cognitive state by one gradient-like step."""

        if isinstance(external_drive, (int, float)):
            drive = [float(external_drive)] * self.n
        else:
            drive = [float(value) for value in external_drive]
            if len(drive) != self.n:
                raise ValueError("external_drive must match n_networks.")

        next_state = []
        for row, row_weights in enumerate(self.H_weights):
            enthalpy_drive = sum(weight * value for weight, value in zip(row_weights, self.X))
            enthalpy_drive += drive[row]

            if regulator.atp < 20.0:
                enthalpy_drive *= regulator.atp / 20.0

            entropy_noise = self.rng.gauss(0.0, regulator.entropy * 0.1)
            chemical_potential = enthalpy_drive - entropy_noise
            next_state.append(math.tanh(chemical_potential / regulator.temperature))

        self.X = next_state
        return self.X.copy()

    def calculate_gibbs_free_energy(self, regulator: MetabolicRegulator) -> float:
        """Calculate ``G = H - T*S`` for the current state."""

        weighted_state = [
            sum(weight * value for weight, value in zip(row_weights, self.X))
            for row_weights in self.H_weights
        ]
        enthalpy = -0.5 * sum(value * weighted for value, weighted in zip(self.X, weighted_state))
        information_entropy = -sum(
            abs(value) * math.log(abs(value) + 1e-9)
            for value in self.X
        )
        return enthalpy - regulator.temperature * (regulator.entropy + information_entropy)


@dataclass(frozen=True)
class SimulationResult:
    """Container returned by ``run_circadian_simulation``."""

    time_hours: list[float]
    atp: list[float]
    entropy: list[float]
    temperature: list[float]
    gibbs_free_energy: list[float]
    network_states: list[list[float]]
    networks: tuple[str, ...]

    @property
    def average_atp(self) -> float:
        return sum(self.atp) / len(self.atp)

    @property
    def max_entropy(self) -> float:
        return max(self.entropy)

    @property
    def gibbs_variance(self) -> float:
        mean = sum(self.gibbs_free_energy) / len(self.gibbs_free_energy)
        return sum((value - mean) ** 2 for value in self.gibbs_free_energy) / len(self.gibbs_free_energy)


def run_circadian_simulation(
    sleep_start_hour: int = 22,
    sleep_duration_hours: int = 8,
    simulation_duration_hours: float = 48.0,
    dt: float = 0.1,
    n_networks: int = 8,
    seed: int | None = 7,
    profile: CircadianProfile | None = None,
) -> SimulationResult:
    """Run a circadian-metabolic cognitive landscape simulation."""

    if sleep_duration_hours < 0 or sleep_duration_hours > 24:
        raise ValueError("sleep_duration_hours must be between 0 and 24.")
    if simulation_duration_hours <= 0:
        raise ValueError("simulation_duration_hours must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    sleep_end_hour = (sleep_start_hour + sleep_duration_hours) % 24
    rng = random.Random(seed)
    regulator = MetabolicRegulator()
    cognitive_state = ThermodynamicCognitiveState(n_networks=n_networks, rng=rng)
    num_steps = int(simulation_duration_hours / dt)

    time_points: list[float] = []
    atp_levels: list[float] = []
    entropy_levels: list[float] = []
    temperature_levels: list[float] = []
    gibbs_levels: list[float] = []
    network_states: list[list[float]] = []

    for step in range(num_steps):
        current_time = step * dt
        current_hour = int(current_time) % 24
        circadian_factor = get_circadian_factor(
            current_hour,
            sleep_start_hour=sleep_start_hour,
            sleep_end_hour=sleep_end_hour,
            profile=profile,
        )
        external_drive = [rng.uniform(-0.1, 0.1) * circadian_factor for _ in range(n_networks)]
        state = cognitive_state.update_state(regulator, external_drive=external_drive)
        regulator.perform_metabolic_cycle(
            state,
            current_hour=current_hour,
            sleep_start_hour=sleep_start_hour,
            sleep_end_hour=sleep_end_hour,
            dt=dt,
            profile=profile,
        )

        time_points.append(current_time)
        atp_levels.append(regulator.atp)
        entropy_levels.append(regulator.entropy)
        temperature_levels.append(regulator.temperature)
        gibbs_levels.append(cognitive_state.calculate_gibbs_free_energy(regulator))
        network_states.append(state)

    return SimulationResult(
        time_hours=time_points,
        atp=atp_levels,
        entropy=entropy_levels,
        temperature=temperature_levels,
        gibbs_free_energy=gibbs_levels,
        network_states=network_states,
        networks=tuple(cognitive_state.networks),
    )


def run_sleep_duration_sweep(
    sleep_durations: Iterable[int] = (4, 6, 8, 10),
    sleep_start_hour: int = 22,
    simulation_duration_hours: float = 48.0,
    dt: float = 0.1,
    seed: int | None = 7,
) -> dict[str, list[float]]:
    """Compare summary metrics across sleep-duration scenarios."""

    results = {
        "sleep_duration": [],
        "avg_atp": [],
        "max_entropy": [],
        "gibbs_variance": [],
    }

    for index, duration in enumerate(sleep_durations):
        result = run_circadian_simulation(
            sleep_start_hour=sleep_start_hour,
            sleep_duration_hours=int(duration),
            simulation_duration_hours=simulation_duration_hours,
            dt=dt,
            seed=None if seed is None else seed + index,
        )
        results["sleep_duration"].append(float(duration))
        results["avg_atp"].append(result.average_atp)
        results["max_entropy"].append(result.max_entropy)
        results["gibbs_variance"].append(result.gibbs_variance)

    return results


def sample_gibbs_surface(
    cognitive_state: ThermodynamicCognitiveState,
    regulator: MetabolicRegulator,
    first_dimension: int = 0,
    second_dimension: int = 1,
    resolution: int = 50,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Sample a two-dimensional Gibbs free-energy slice for visualization."""

    if resolution < 2:
        raise ValueError("resolution must be at least 2.")
    if first_dimension == second_dimension:
        raise ValueError("dimensions must be distinct.")
    if not (0 <= first_dimension < cognitive_state.n and 0 <= second_dimension < cognitive_state.n):
        raise ValueError("dimensions must be valid network indices.")

    values = [-1.0 + 2.0 * index / (resolution - 1) for index in range(resolution)]
    x_grid = [[x for x in values] for _ in values]
    y_grid = [[y for _ in values] for y in values]
    z_gibbs = [[0.0 for _ in values] for _ in values]
    original_state = cognitive_state.X.copy()

    try:
        for row, y_value in enumerate(values):
            for column, x_value in enumerate(values):
                temp_state = original_state.copy()
                temp_state[first_dimension] = x_value
                temp_state[second_dimension] = y_value
                cognitive_state.X = temp_state
                z_gibbs[row][column] = cognitive_state.calculate_gibbs_free_energy(regulator)
    finally:
        cognitive_state.X = original_state

    return x_grid, y_grid, z_gibbs


if __name__ == "__main__":  # pragma: no cover - example usage
    simulation = run_circadian_simulation()
    print("Simulation complete")
    print(f"Average ATP: {simulation.average_atp:.2f}")
    print(f"Max entropy: {simulation.max_entropy:.2f}")
    print(f"Gibbs variance: {simulation.gibbs_variance:.2f}")

    sweep = run_sleep_duration_sweep()
    for duration, avg_atp, max_entropy, gibbs_variance in zip(
        sweep["sleep_duration"],
        sweep["avg_atp"],
        sweep["max_entropy"],
        sweep["gibbs_variance"],
    ):
        print(
            f"Sleep {duration:.0f}h: avg ATP={avg_atp:.2f}, "
            f"max entropy={max_entropy:.2f}, G variance={gibbs_variance:.2f}"
        )
