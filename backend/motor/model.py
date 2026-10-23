"""Separate motor dynamics for converting steering signals into movement state.

Purpose: turn decoded left/right/forward commands into yaw, forward velocity,
and position updates for a future embodied fly.
Why it exists: the neural circuit and steering decoder must not directly mutate
an agent transform; motor dynamics are an independent, testable boundary.
Inputs: bounded steering command values and a scientific timestep.
Outputs: immutable ``MotorState`` snapshots containing position, yaw, velocity,
and yaw rate.
Dependencies: Python standard library and NumPy.

Coordinate convention
---------------------
The model follows the navigation convention: zero yaw points along -Z, positive
turn command (right > left) produces positive yaw, and Y remains unchanged in
this phase. Position is integrated using the new yaw and updated forward speed.
The state is not a Godot transform and the model performs no collision handling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from backend.navigation.decoder import SteeringCommand
from backend.navigation.math import heading_vector_from_yaw


@dataclass(frozen=True, slots=True)
class MotorParameters:
    """Physical/computational bounds for the student-scale motor model."""

    time_step: float = 0.01
    maximum_forward_speed: float = 2.0
    maximum_yaw_rate: float = math.pi
    forward_acceleration: float = 4.0
    forward_deceleration: float = 6.0

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive dynamic limits."""
        for name in (
            "time_step",
            "maximum_forward_speed",
            "maximum_yaw_rate",
            "forward_acceleration",
            "forward_deceleration",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class MotorState:
    """Immutable post-step movement state returned by ``MotorModel.step``."""

    position: tuple[float, float, float]
    yaw_radians: float
    yaw_rate: float
    forward_velocity: float
    step_index: int

    @property
    def speed(self) -> float:
        """Return scalar forward speed for dashboards and metrics."""
        return self.forward_velocity


class MotorModel:
    """Integrate steering commands into movement state without renderer access."""

    def __init__(
        self,
        parameters: MotorParameters | None = None,
        *,
        initial_position: Sequence[float] = (0.0, 0.0, 0.0),
        initial_yaw_radians: float = 0.0,
    ) -> None:
        """Create a motor state with validated position and heading."""
        self.parameters = parameters or MotorParameters()
        self._initial_position = _position_tuple(initial_position)
        if not math.isfinite(initial_yaw_radians):
            raise ValueError("initial_yaw_radians must be finite")
        self.position = self._initial_position
        self.yaw_radians = _wrap_yaw(initial_yaw_radians)
        self.yaw_rate = 0.0
        self.forward_velocity = 0.0
        self.step_index = 0

    def reset(
        self,
        *,
        position: Sequence[float] | None = None,
        yaw_radians: float | None = None,
    ) -> MotorState:
        """Reset movement state for a reproducible trial and return its snapshot."""
        next_position = self._initial_position if position is None else _position_tuple(position)
        next_yaw = 0.0 if yaw_radians is None else yaw_radians
        if not math.isfinite(next_yaw):
            raise ValueError("yaw_radians must be finite")
        self.position = next_position
        self.yaw_radians = _wrap_yaw(next_yaw)
        self.yaw_rate = 0.0
        self.forward_velocity = 0.0
        self.step_index = 0
        return self.state

    @property
    def state(self) -> MotorState:
        """Return the current immutable movement snapshot."""
        return MotorState(
            position=self.position,
            yaw_radians=self.yaw_radians,
            yaw_rate=self.yaw_rate,
            forward_velocity=self.forward_velocity,
            step_index=self.step_index,
        )

    def step(
        self,
        command: SteeringCommand | Mapping[str, float],
        *,
        time_step: float | None = None,
    ) -> MotorState:
        """Apply one steering command and integrate yaw, velocity, and position.

        WHAT: validate left/right/forward signals, derive differential yaw rate,
        rate-limit forward speed toward the requested speed, then integrate
        position along the updated horizontal heading.
        WHY: the motor model is the separate brain-to-body dynamics boundary.
        OUTPUT: post-step ``MotorState``; no external transform is mutated.
        """
        dt = self.parameters.time_step if time_step is None else time_step
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("time_step must be finite and positive")
        signals = _command_mapping(command)
        left = _signal(signals, "left")
        right = _signal(signals, "right")
        forward = _signal(signals, "forward")

        self.yaw_rate = (right - left) * self.parameters.maximum_yaw_rate
        self.yaw_radians = _wrap_yaw(self.yaw_radians + self.yaw_rate * dt)
        target_speed = forward * self.parameters.maximum_forward_speed
        if target_speed >= self.forward_velocity:
            max_delta = self.parameters.forward_acceleration * dt
        else:
            max_delta = self.parameters.forward_deceleration * dt
        delta_velocity = float(np.clip(
            target_speed - self.forward_velocity,
            -max_delta,
            max_delta,
        ))
        self.forward_velocity = float(self.forward_velocity + delta_velocity)
        heading = heading_vector_from_yaw(self.yaw_radians)
        displacement = heading * self.forward_velocity * dt
        self.position = (
            self.position[0] + float(displacement[0]),
            self.position[1],
            self.position[2] + float(displacement[2]),
        )
        self.step_index += 1
        return self.state


def _command_mapping(command: SteeringCommand | Mapping[str, float]) -> Mapping[str, float]:
    """Accept the decoder object or a motor-model-ready signal mapping."""
    if isinstance(command, SteeringCommand):
        return command.as_dict()
    if not isinstance(command, Mapping):
        raise TypeError("command must be SteeringCommand or a mapping")
    return command


def _signal(command: Mapping[str, float], name: str) -> float:
    """Validate one normalized steering signal in [0, 1]."""
    if name not in command:
        raise KeyError(f"motor command missing signal: {name}")
    value = command[name]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"motor signal {name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"motor signal {name} must be in [0, 1]")
    return float(value)


def _position_tuple(position: Sequence[float]) -> tuple[float, float, float]:
    """Validate and normalize a 3D position without exposing mutable arrays."""
    if len(position) != 3:
        raise ValueError("position must contain exactly three coordinates")
    values = tuple(float(value) for value in position)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("position coordinates must be finite")
    return values


def _wrap_yaw(yaw_radians: float) -> float:
    """Wrap yaw to [-pi, pi) for stable logging and dashboard display."""
    return (yaw_radians + math.pi) % (2.0 * math.pi) - math.pi
