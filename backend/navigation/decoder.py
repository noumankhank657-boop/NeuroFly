"""Decode LIF circuit activity into continuous steering signals.

Purpose: translate selected circuit-node activity into interpretable left, right,
and forward command values for the future motor model.
Why it exists: neural activity must pass through a separate, inspectable decoder;
the circuit must not directly manipulate a 3D transform.
Inputs: a ``ToyNeuralCircuit`` with a completed step, explicit output-node
mapping, and activity/command parameters.
Outputs: bounded continuous ``SteeringCommand`` values plus raw channel activity.
Dependencies: Python standard library and the local LIF circuit.

Activity is continuous even between spikes: each node contributes a weighted
combination of its transient spike flag and normalized subthreshold membrane
potential. Population channels aggregate their mapped nodes by mean. The decoder
performs no movement, route planning, learning, or biological inference.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import math

from backend.neurons import ToyNeuralCircuit


@dataclass(frozen=True, slots=True)
class SteeringOutputNodes:
    """Map steering channels to one or more explicit LIF output node IDs."""

    left: tuple[str, ...]
    right: tuple[str, ...]
    forward: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require non-empty, distinct channel populations."""
        channels = (self.left, self.right, self.forward)
        if any(not channel for channel in channels):
            raise ValueError("each steering channel needs at least one node")
        nodes = [node for channel in channels for node in channel]
        if any(not isinstance(node, str) or not node.strip() for node in nodes):
            raise ValueError("steering node IDs must be non-empty strings")
        if len(nodes) != len(set(nodes)):
            raise ValueError("steering channel node populations must not overlap")

    @classmethod
    def from_names(cls, left: str, right: str, forward: str) -> "SteeringOutputNodes":
        """Create single-node channel mapping without hiding node identity."""
        return cls((left,), (right,), (forward,))


@dataclass(frozen=True, slots=True)
class SteeringDecoderParameters:
    """Weights and gains for continuous activity-to-command decoding."""

    spike_weight: float = 0.7
    subthreshold_weight: float = 0.3
    left_gain: float = 1.0
    right_gain: float = 1.0
    forward_gain: float = 1.0
    maximum_signal: float = 1.0
    # Number of most-recent circuit steps averaged into one command. 1 keeps the
    # original single-step behaviour. A larger window turns on/off spike flicker
    # into a firing-RATE estimate (a window of N steps = N * dt seconds).
    rate_window_steps: int = 1

    def __post_init__(self) -> None:
        """Validate that activity weights and signal gains are finite and useful."""
        if not isinstance(self.rate_window_steps, int) or self.rate_window_steps < 1:
            raise ValueError("rate_window_steps must be an integer >= 1")
        if not math.isfinite(self.spike_weight) or self.spike_weight < 0.0:
            raise ValueError("spike_weight must be finite and non-negative")
        if not math.isfinite(self.subthreshold_weight) or self.subthreshold_weight < 0.0:
            raise ValueError("subthreshold_weight must be finite and non-negative")
        if self.spike_weight + self.subthreshold_weight <= 0.0:
            raise ValueError("at least one activity weight must be positive")
        for name in ("left_gain", "right_gain", "forward_gain", "maximum_signal"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.maximum_signal <= 0.0:
            raise ValueError("maximum_signal must be positive")


@dataclass(frozen=True, slots=True)
class SteeringCommand:
    """Continuous decoded steering output and its raw channel diagnostics."""

    left: float
    right: float
    forward: float
    raw_left_activity: float
    raw_right_activity: float
    raw_forward_activity: float
    step_index: int

    def as_dict(self) -> dict[str, float]:
        """Return only the command channels in motor-model-ready form."""
        return {"left": self.left, "right": self.right, "forward": self.forward}


class SteeringDecoder:
    """Decode selected LIF population activity without moving an agent."""

    def __init__(
        self,
        output_nodes: SteeringOutputNodes,
        parameters: SteeringDecoderParameters | None = None,
    ) -> None:
        """Create a decoder with explicit output populations and gains."""
        self.output_nodes = output_nodes
        self.parameters = parameters or SteeringDecoderParameters()
        self._history: deque[tuple[float, float, float]] = deque(
            maxlen=self.parameters.rate_window_steps
        )

    def reset(self) -> None:
        """Forget the rate-window history (call when the circuit is reset)."""
        self._history.clear()

    def decode(self, circuit: ToyNeuralCircuit) -> SteeringCommand:
        """Decode the circuit's most recent completed step into continuous signals.

        WHAT: normalize each mapped neuron's membrane potential between rest and
        threshold, combine it with its spike flag, average each channel, apply
        channel gains, and clamp the command range.
        WHY: this keeps the brain-to-command boundary interpretable and separate
        from the future motor transform update.
        OUTPUT: bounded ``SteeringCommand`` with raw activity diagnostics.
        """
        result = circuit.last_result
        if result is None:
            raise ValueError("circuit must be stepped before decoding")
        all_nodes = set(circuit.neurons)
        requested = [node for channel in self._channels() for node in channel]
        missing = sorted(set(requested) - all_nodes)
        if missing:
            raise KeyError(f"decoder output nodes are absent from circuit: {missing}")

        self._history.append((
            self._population_activity(circuit, self.output_nodes.left),
            self._population_activity(circuit, self.output_nodes.right),
            self._population_activity(circuit, self.output_nodes.forward),
        ))
        n = len(self._history)
        raw_left = sum(h[0] for h in self._history) / n
        raw_right = sum(h[1] for h in self._history) / n
        raw_forward = sum(h[2] for h in self._history) / n
        left = _bounded(raw_left * self.parameters.left_gain, self.parameters.maximum_signal)
        right = _bounded(raw_right * self.parameters.right_gain, self.parameters.maximum_signal)
        forward = _bounded(raw_forward * self.parameters.forward_gain, self.parameters.maximum_signal)
        return SteeringCommand(
            left=left,
            right=right,
            forward=forward,
            raw_left_activity=raw_left,
            raw_right_activity=raw_right,
            raw_forward_activity=raw_forward,
            step_index=result.step_index,
        )

    def _population_activity(
        self,
        circuit: ToyNeuralCircuit,
        node_names: Iterable[str],
    ) -> float:
        """Aggregate weighted continuous activities across one output population."""
        values = []
        weight_total = self.parameters.spike_weight + self.parameters.subthreshold_weight
        for name in node_names:
            neuron = circuit.neurons[name]
            threshold_span = neuron.parameters.threshold_potential - neuron.parameters.resting_potential
            normalized_potential = (
                (neuron.membrane_potential - neuron.parameters.resting_potential)
                / threshold_span
            )
            normalized_potential = _bounded(normalized_potential, 1.0)
            activity = (
                self.parameters.spike_weight * float(neuron.spike)
                + self.parameters.subthreshold_weight * normalized_potential
            ) / weight_total
            values.append(_bounded(activity, 1.0))
        return sum(values) / len(values)

    def _channels(self) -> tuple[tuple[str, ...], ...]:
        """Return channel populations in a stable validation order."""
        return (self.output_nodes.left, self.output_nodes.right, self.output_nodes.forward)


def _bounded(value: float, maximum: float) -> float:
    """Clamp a finite scalar into the inclusive interval [0, maximum]."""
    if not math.isfinite(value):
        raise ValueError("decoder activity must be finite")
    return min(max(value, 0.0), maximum)
