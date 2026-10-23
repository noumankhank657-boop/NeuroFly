"""Unit tests for Phase 2 navigation geometry.

Purpose: verify vector, heading, goal, proximity, and local cue primitives.
Why it exists: these functions become the scientific contract for later sensory
encoding and should fail loudly on invalid geometry.
Inputs: deterministic numeric fixtures.
Outputs: standard-library unittest results.
Dependencies: NumPy and the project navigation package.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from backend.navigation.math import (
    heading_vector_from_yaw,
    left_right_environmental_cues,
    normalize_vector,
    obstacle_proximity,
    relative_goal_direction,
    signed_heading_error,
)


class NavigationMathTests(unittest.TestCase):
    """Check expected geometry and bounded environmental cue behavior."""

    def test_normalize_vector(self) -> None:
        result = normalize_vector([3.0, 0.0, 4.0])
        np.testing.assert_allclose(result, [0.6, 0.0, 0.8])
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_normalize_rejects_zero_and_nonfinite_vectors(self) -> None:
        with self.assertRaises(ValueError):
            normalize_vector([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            normalize_vector([math.nan, 0.0])

    def test_yaw_zero_points_along_negative_z(self) -> None:
        np.testing.assert_allclose(heading_vector_from_yaw(0.0), [0.0, 0.0, -1.0])

    def test_heading_error_sign_and_magnitude(self) -> None:
        heading = heading_vector_from_yaw(0.0)
        self.assertAlmostEqual(signed_heading_error(heading, [1.0, 0.0, 0.0]), math.pi / 2)
        self.assertAlmostEqual(signed_heading_error(heading, [-1.0, 0.0, 0.0]), -math.pi / 2)
        self.assertAlmostEqual(signed_heading_error(heading, [0.0, 0.0, -2.0]), 0.0)

    def test_relative_goal_direction(self) -> None:
        result = relative_goal_direction([2.0, 1.0, 3.0], [2.0, 4.0, 7.0])
        np.testing.assert_allclose(result, [0.0, 3.0 / 5.0, 4.0 / 5.0])
        with self.assertRaises(ValueError):
            relative_goal_direction([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_obstacle_proximity_is_bounded(self) -> None:
        self.assertEqual(obstacle_proximity(0.0, 10.0), 1.0)
        self.assertEqual(obstacle_proximity(5.0, 10.0), 0.5)
        self.assertEqual(obstacle_proximity(20.0, 10.0), 0.0)
        with self.assertRaises(ValueError):
            obstacle_proximity(1.0, 0.0)

    def test_environmental_cues_split_left_and_right(self) -> None:
        left, right = left_right_environmental_cues(
            [0.0, 0.0, 0.0],
            heading_vector_from_yaw(0.0),
            [[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0]],
            sensing_radius=10.0,
        )
        self.assertGreater(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(left, right)
        self.assertTrue(0.0 <= left <= 1.0)
        self.assertTrue(0.0 <= right <= 1.0)

    def test_behind_obstacles_are_not_environmental_cues(self) -> None:
        cues = left_right_environmental_cues(
            [0.0, 0.0, 0.0],
            heading_vector_from_yaw(0.0),
            [[-2.0, 0.0, 2.0], [2.0, 0.0, 2.0]],
            sensing_radius=10.0,
        )
        self.assertEqual(cues, (0.0, 0.0))

    def test_pure_calculation_is_reproducible(self) -> None:
        args = ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [[1.0, 0.0, -3.0]])
        first = left_right_environmental_cues(*args, sensing_radius=5.0)
        second = left_right_environmental_cues(*args, sensing_radius=5.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
