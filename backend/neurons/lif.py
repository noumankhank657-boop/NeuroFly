"""Understandable Leaky Integrate-and-Fire neuron dynamics.

Purpose: provide the smallest inspectable spiking unit for the NueroFly
controller.
Why it exists: later circuit code needs a documented neuron with membrane
potential, input current, leak, threshold, reset, and spike state rather than a
black-box neural computation.
Inputs: scalar input current and a scientific simulation timestep.
Outputs: a per-step ``LIFStepResult`` plus inspectable neuron state.
Dependencies: Python standard library only.

Model
-----
The membrane follows a forward-Euler discretisation of the leaky integrator:

    dV/dt = ((V_rest - V) + R_m * I) / tau_m

After integration, a spike occurs when ``V >= V_threshold``. The potential is
then reset to ``V_reset``. An optional absolute refractory period holds the
potential at reset and suppresses additional spikes while it counts down. The
parameters are computational abstractions; they are not claims of biological
exactness or a complete Drosophila neuron model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class LIFParameters:
    """Validated scalar parameters for one current-based LIF neuron.

    ``resistance`` and current use consistent abstract units. Potentials and
    time are represented as floating-point values; callers should keep one unit
    system across the simulation.
    """

    membrane_time_constant: float = 0.02
    resting_potential: float = 0.0
    reset_potential: float = 0.0
    threshold_potential: float = 1.0
    membrane_resistance: float = 1.0
    time_step: float = 0.001
    refractory_period: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid physical/computational settings early."""
        positive_fields = {
            "membrane_time_constant": self.membrane_time_constant,
            "time_step": self.time_step,
            "membrane_resistance": self.membrane_resistance,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.refractory_period) or self.refractory_period < 0.0:
            raise ValueError("refractory_period must be finite and non-negative")
        for name, value in {
            "resting_potential": self.resting_potential,
            "reset_potential": self.reset_potential,
            "threshold_potential": self.threshold_potential,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.threshold_potential <= self.reset_potential:
            raise ValueError("threshold_potential must be above reset_potential")


@dataclass(frozen=True, slots=True)
class LIFStepResult:
    """Immutable output of one LIF update.

    ``membrane_potential`` is the post-update value. A spiking step reports
    ``spike=True`` even though the stored potential has already been reset.
    """

    membrane_potential: float
    input_current: float
    spike: bool
    refractory_remaining: float


class LIFNeuron:
    """A single deterministic current-based LIF neuron.

    The class intentionally has no weights, graph edges, sensory interpretation,
    or motor output. Those responsibilities belong to later modules.
    """

    def __init__(
        self,
        parameters: LIFParameters | None = None,
        *,
        initial_potential: float | None = None,
    ) -> None:
        """Create a neuron with validated parameters and an initial state."""
        self.parameters = parameters or LIFParameters()
        potential = (
            self.parameters.resting_potential
            if initial_potential is None
            else initial_potential
        )
        if not math.isfinite(potential):
            raise ValueError("initial_potential must be finite")
        self.membrane_potential = float(potential)
        self.input_current = 0.0
        self.spike = False
        self.spike_count = 0
        self.refractory_remaining = 0.0

    def reset(self, *, potential: float | None = None) -> None:
        """Reset dynamic state for a new trial or repeated deterministic run."""
        reset_value = self.parameters.reset_potential if potential is None else potential
        if not math.isfinite(reset_value):
            raise ValueError("potential must be finite")
        self.membrane_potential = float(reset_value)
        self.input_current = 0.0
        self.spike = False
        self.spike_count = 0
        self.refractory_remaining = 0.0

    def step(
        self,
        input_current: float,
        *,
        time_step: float | None = None,
    ) -> LIFStepResult:
        """Advance the neuron by one timestep and return its observable result.

        WHAT: apply leak and input integration, detect threshold crossing, reset,
        and update refractory state.
        WHY: this is the complete transparent computational rule for one LIF
        neuron; the caller can inspect every state needed by later visualization.
        OUTPUT: post-step potential, input current, spike flag, and refractory
        time remaining.
        """
        if not math.isfinite(input_current):
            raise ValueError("input_current must be finite")
        dt = self.parameters.time_step if time_step is None else time_step
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("time_step must be finite and positive")

        self.input_current = float(input_current)
        self.spike = False

        if self.refractory_remaining > _EPSILON:
            self.membrane_potential = self.parameters.reset_potential
            self.refractory_remaining = max(0.0, self.refractory_remaining - dt)
            return self._result()

        leak_and_drive = (
            (self.parameters.resting_potential - self.membrane_potential)
            + self.parameters.membrane_resistance * self.input_current
        )
        self.membrane_potential += (
            dt / self.parameters.membrane_time_constant
        ) * leak_and_drive

        if self.membrane_potential >= self.parameters.threshold_potential:
            self.spike = True
            self.spike_count += 1
            self.membrane_potential = self.parameters.reset_potential
            self.refractory_remaining = self.parameters.refractory_period

        return self._result()

    def _result(self) -> LIFStepResult:
        """Build an immutable snapshot without exposing mutable internals."""
        return LIFStepResult(
            membrane_potential=self.membrane_potential,
            input_current=self.input_current,
            spike=self.spike,
            refractory_remaining=self.refractory_remaining,
        )
