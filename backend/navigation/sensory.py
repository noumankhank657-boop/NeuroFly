"""Encode local goal and obstacle geometry into interpretable circuit currents.

Purpose: convert Phase 2 geometric cues into external currents for selected
sensory/input nodes in a Phase 6 LIF circuit.
Why it exists: the neural circuit should receive directional and environmental
information derived from the current state, not a route, waypoint, or hidden
planner output.
Inputs: current position/heading, goal position, obstacle positions, explicit
channel-to-node mapping, and encoding gains.
Outputs: bounded left, right, and forward currents plus an auditable observation.
Dependencies: NumPy and local navigation geometry helpers.

The encoder is a hand-specified computational sensory model. It is not trained,
does not infer a path, and does not claim to reproduce a fly's exact sensory
system. A right obstacle increases left steering current and vice versa; forward
current is reduced by nearby local obstacles.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .math import (
    ArrayLike,
    horizontal_vector,
    left_right_environmental_cues,
    signed_heading_error,
)


@dataclass(frozen=True, slots=True)
class SensoryChannelTargets:
    """Map abstract sensory channels to explicit circuit node identifiers."""

    left: str
    right: str
    forward: str

    def __post_init__(self) -> None:
        """Require three distinct non-empty node names."""
        values = (self.left, self.right, self.forward)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("sensory target names must be non-empty strings")
        if len(set(values)) != 3:
            raise ValueError("left, right, and forward target names must be distinct")


@dataclass(frozen=True, slots=True)
class SensoryEncodingParameters:
    """Gains and bounds for the hand-specified sensory current model."""

    goal_turn_gain: float = 1.0
    forward_gain: float = 1.0
    obstacle_avoidance_gain: float = 1.0
    obstacle_forward_suppression: float = 0.5
    sensing_radius: float = 10.0
    maximum_current: float = 2.0

    def __post_init__(self) -> None:
        """Validate gains so currents remain finite and interpretable."""
        for name in (
            "goal_turn_gain",
            "forward_gain",
            "obstacle_avoidance_gain",
            "sensing_radius",
            "maximum_current",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.sensing_radius <= 0.0 or self.maximum_current <= 0.0:
            raise ValueError("sensing_radius and maximum_current must be positive")
        if not math.isfinite(self.obstacle_forward_suppression) or not 0.0 <= self.obstacle_forward_suppression <= 1.0:
            raise ValueError("obstacle_forward_suppression must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SensoryEncodingResult:
    """Auditable output of one sensory encoding operation."""

    heading_error_radians: float
    goal_distance: float
    goal_left_drive: float
    goal_right_drive: float
    goal_forward_drive: float
    obstacle_left_cue: float
    obstacle_right_cue: float
    currents: dict[str, float]


class NavigationSensoryEncoder:
    """Encode current local geometry into three bounded external currents."""

    def __init__(
        self,
        targets: SensoryChannelTargets,
        parameters: SensoryEncodingParameters | None = None,
    ) -> None:
        """Create an encoder with explicit node targets and computational gains."""
        self.targets = targets
        self.parameters = parameters or SensoryEncodingParameters()

    def encode(
        self,
        current_position: ArrayLike,
        current_heading: ArrayLike,
        goal_position: ArrayLike,
        obstacle_positions: Iterable[ArrayLike] = (),
    ) -> SensoryEncodingResult:
        """Compute goal and obstacle currents from the current local state.

        WHAT: calculate signed goal error, local obstacle cues, and three bounded
        channel currents; map them to the caller's named circuit nodes.
        WHY: this is a transparent state-to-input boundary for the LIF circuit.
        OUTPUT: ``SensoryEncodingResult`` with intermediate cues and node currents.
        """
        current = _validated_position(current_position, "current_position")
        goal = _validated_position(goal_position, "goal_position")
        if current.size != goal.size:
            raise ValueError("current_position and goal_position must have equal dimensions")
        heading = horizontal_vector(current_heading, name="current_heading")
        goal_displacement = goal - current
        goal_distance = float(np.linalg.norm(goal_displacement))
        heading_error = _horizontal_goal_error(current_heading, goal_displacement)

        goal_left_drive = max(0.0, -math.sin(heading_error)) * self.parameters.goal_turn_gain
        goal_right_drive = max(0.0, math.sin(heading_error)) * self.parameters.goal_turn_gain
        goal_forward_drive = max(0.0, math.cos(heading_error)) * self.parameters.forward_gain
        if np.linalg.norm(horizontal_vector(goal_displacement, name="goal_displacement")) <= 1e-12:
            goal_forward_drive = 0.0
            goal_left_drive = 0.0
            goal_right_drive = 0.0

        obstacle_left, obstacle_right = left_right_environmental_cues(
            current,
            heading,
            obstacle_positions,
            sensing_radius=self.parameters.sensing_radius,
        )
        obstacle_strength = max(obstacle_left, obstacle_right)
        left_current = goal_left_drive + obstacle_right * self.parameters.obstacle_avoidance_gain
        right_current = goal_right_drive + obstacle_left * self.parameters.obstacle_avoidance_gain
        forward_current = goal_forward_drive * (
            1.0 - self.parameters.obstacle_forward_suppression * obstacle_strength
        )
        currents = {
            self.targets.left: _bounded_current(left_current, self.parameters.maximum_current),
            self.targets.right: _bounded_current(right_current, self.parameters.maximum_current),
            self.targets.forward: _bounded_current(forward_current, self.parameters.maximum_current),
        }
        return SensoryEncodingResult(
            heading_error_radians=heading_error,
            goal_distance=goal_distance,
            goal_left_drive=goal_left_drive,
            goal_right_drive=goal_right_drive,
            goal_forward_drive=goal_forward_drive,
            obstacle_left_cue=obstacle_left,
            obstacle_right_cue=obstacle_right,
            currents=currents,
        )


def _validated_position(value: ArrayLike, name: str) -> np.ndarray:
    """Validate a position without exposing the geometry module's private helper."""
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size not in (2, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 2D or 3D vector")
    return result


def _horizontal_goal_error(current_heading: ArrayLike, displacement: np.ndarray) -> float:
    """Return zero for a vertically aligned/coincident goal, otherwise its angle."""
    horizontal_goal = horizontal_vector(displacement, name="goal_displacement")
    if float(np.linalg.norm(horizontal_goal)) <= 1e-12:
        horizontal_heading = horizontal_vector(current_heading, name="current_heading")
        if float(np.linalg.norm(horizontal_heading)) <= 1e-12:
            raise ValueError("current_heading must have a non-zero horizontal component")
        return 0.0
    return signed_heading_error(current_heading, displacement)


def _bounded_current(value: float, maximum: float) -> float:
    """Clamp a non-negative sensory drive to the configured current range."""
    return float(np.clip(value, 0.0, maximum))
