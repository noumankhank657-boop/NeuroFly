"""Navigation geometry primitives for the NueroFly prototype.

Purpose: convert positions, headings, goals, and simple obstacle observations into
well-defined numerical cues for later sensory encoding.
Why it exists: Phase 2 isolates geometric reasoning from neurons, motor dynamics,
and Godot transforms so each assumption can be tested independently.
Inputs: NumPy-compatible 2D/3D vectors, positions, and scalar sensor settings.
Outputs: normalized vectors, signed angular errors, proximity strengths, and
left/right environmental cue strengths.
Dependencies: NumPy.

Coordinate convention
---------------------
The model uses a right-handed 3D frame with Y as vertical. Navigation geometry
projects vectors onto the horizontal X-Z plane. A heading is a non-zero direction
vector, not an absolute yaw angle. Signed angles use the horizontal cross-product
orientation: positive means the target/obstacle is to the right of the heading,
negative means it is to the left. This sign convention is explicit because the
future motor model will map it to yaw direction.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np


ArrayLike = np.ndarray | Iterable[float]
_EPSILON = 1e-12


def _as_vector(vector: ArrayLike, *, name: str) -> np.ndarray:
    """Convert an input to a finite one-dimensional floating-point vector."""
    result = np.asarray(vector, dtype=float)
    if result.ndim != 1 or result.size not in (2, 3):
        raise ValueError(f"{name} must be a finite 2D or 3D vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def normalize_vector(vector: ArrayLike, *, name: str = "vector") -> np.ndarray:
    """Return a unit vector while rejecting zero-length or invalid inputs.

    WHAT: divide a vector by its Euclidean norm.
    WHY: downstream direction and angle calculations require scale-independent
    directions; silently normalizing a zero vector would hide invalid state.
    OUTPUT: a new floating-point vector with norm approximately one.
    """
    value = _as_vector(vector, name=name)
    norm = float(np.linalg.norm(value))
    if norm <= _EPSILON:
        raise ValueError(f"{name} must have non-zero length")
    return value / norm


def horizontal_vector(vector: ArrayLike, *, name: str = "vector") -> np.ndarray:
    """Project a 2D/3D vector onto the horizontal X-Z plane as a 2D array."""
    value = _as_vector(vector, name=name)
    if value.size == 2:
        return value.copy()
    return np.array([value[0], value[2]], dtype=float)


def heading_vector_from_yaw(yaw_radians: float) -> np.ndarray:
    """Create a 3D horizontal heading from yaw, using -Z as zero-yaw forward."""
    if not math.isfinite(yaw_radians):
        raise ValueError("yaw_radians must be finite")
    return np.array([math.sin(yaw_radians), 0.0, -math.cos(yaw_radians)])


def signed_heading_error(
    current_heading: ArrayLike,
    target_direction: ArrayLike,
) -> float:
    """Return the signed smallest horizontal angle from heading to target.

    WHAT: compare two horizontal direction vectors with dot/cross products.
    WHY: the brain needs directional error, not a perfect route or transformed
    target position. The result is bounded and independent of vector magnitude.
    OUTPUT: radians in [-pi, pi], positive for target-right and negative for
    target-left under the project convention.
    """
    heading = normalize_vector(horizontal_vector(current_heading, name="current_heading"), name="current_heading")
    target = normalize_vector(horizontal_vector(target_direction, name="target_direction"), name="target_direction")
    dot = float(np.clip(np.dot(heading, target), -1.0, 1.0))
    cross_y = float(heading[0] * target[1] - heading[1] * target[0])
    return math.atan2(cross_y, dot)


def relative_goal_direction(current_position: ArrayLike, goal_position: ArrayLike) -> np.ndarray:
    """Return the normalized direction from the current position to the goal."""
    current = _as_vector(current_position, name="current_position")
    goal = _as_vector(goal_position, name="goal_position")
    if current.size != goal.size:
        raise ValueError("current_position and goal_position must have equal dimensions")
    return normalize_vector(goal - current, name="goal displacement")


def obstacle_proximity(distance: float, sensing_radius: float) -> float:
    """Map obstacle distance to a bounded proximity strength in [0, 1].

    WHAT: linearly encode zero distance as one and the sensing radius as zero.
    WHY: this is an interpretable local cue, not a collision-free path planner.
    OUTPUT: a scalar suitable for later sensory input; distances beyond the
    sensing radius produce zero and negative distances are treated as contact.
    """
    if not math.isfinite(distance) or not math.isfinite(sensing_radius):
        raise ValueError("distance and sensing_radius must be finite")
    if sensing_radius <= 0.0:
        raise ValueError("sensing_radius must be positive")
    return float(np.clip(1.0 - max(distance, 0.0) / sensing_radius, 0.0, 1.0))


def left_right_environmental_cues(
    current_position: ArrayLike,
    current_heading: ArrayLike,
    obstacle_positions: Iterable[ArrayLike],
    *,
    sensing_radius: float,
    side_angle_limit_radians: float = math.pi / 2.0,
) -> tuple[float, float]:
    """Aggregate local obstacle strengths into interpretable left/right cues.

    WHAT: weight each visible obstacle by proximity and split it by signed side
    relative to the current heading; obstacles behind the fly are excluded.
    WHY: the future circuit needs directional environmental information without
    being handed a shortest path or global map.
    OUTPUT: ``(left_cue, right_cue)`` in [0, 1]. Multiple obstacles combine with
    a saturating rule so cues remain bounded.
    """
    if not math.isfinite(side_angle_limit_radians) or not 0.0 < side_angle_limit_radians <= math.pi:
        raise ValueError("side_angle_limit_radians must be in (0, pi]")
    if not math.isfinite(sensing_radius) or sensing_radius <= 0.0:
        raise ValueError("sensing_radius must be positive and finite")

    position = _as_vector(current_position, name="current_position")
    heading = normalize_vector(horizontal_vector(current_heading, name="current_heading"), name="current_heading")
    left_strength = 0.0
    right_strength = 0.0

    for index, obstacle in enumerate(obstacle_positions):
        obstacle_value = _as_vector(obstacle, name=f"obstacle_positions[{index}]")
        if obstacle_value.size != position.size:
            raise ValueError("all obstacle positions must match current_position dimensions")
        displacement = horizontal_vector(obstacle_value - position, name="obstacle displacement")
        distance = float(np.linalg.norm(displacement))
        if distance <= _EPSILON:
            # Contact is an equally strong local cue on both sides.
            strength = 1.0
            left_strength = 1.0
            right_strength = 1.0
            continue
        direction = displacement / distance
        angle = signed_heading_error(heading, direction)
        if abs(angle) > side_angle_limit_radians:
            continue
        strength = obstacle_proximity(distance, sensing_radius)
        if angle < 0.0:
            left_strength = 1.0 - (1.0 - left_strength) * (1.0 - strength)
        else:
            right_strength = 1.0 - (1.0 - right_strength) * (1.0 - strength)

    return float(left_strength), float(right_strength)
