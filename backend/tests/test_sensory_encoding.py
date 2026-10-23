"""Tests for Phase 7 goal and obstacle sensory-current encoding.

Purpose: verify local geometry becomes bounded, interpretable circuit currents.
Why it exists: sensory encoding must be deterministic and auditable before it is
connected to a neural decoder or motor model.
Inputs: fixed positions, headings, goals, obstacles, and encoding parameters.
Outputs: standard-library unittest results.
Dependencies: NumPy and local navigation modules.
"""

from __future__ import annotations

import math
import unittest

from backend.navigation import (
    NavigationSensoryEncoder,
    SensoryChannelTargets,
    SensoryEncodingParameters,
    heading_vector_from_yaw,
)


class SensoryEncodingTests(unittest.TestCase):
    """Check directional goal drives and local obstacle effects."""

    def setUp(self) -> None:
        self.encoder = NavigationSensoryEncoder(
            SensoryChannelTargets("left_node", "right_node", "forward_node"),
            SensoryEncodingParameters(
                goal_turn_gain=1.0,
                forward_gain=1.0,
                obstacle_avoidance_gain=1.0,
                obstacle_forward_suppression=0.5,
                sensing_radius=10.0,
                maximum_current=2.0,
            ),
        )
        self.position = [0.0, 0.0, 0.0]
        self.heading = heading_vector_from_yaw(0.0)

    def test_forward_goal_drives_only_forward_channel(self) -> None:
        result = self.encoder.encode(self.position, self.heading, [0.0, 0.0, -5.0])
        self.assertAlmostEqual(result.heading_error_radians, 0.0)
        self.assertEqual(result.currents["left_node"], 0.0)
        self.assertEqual(result.currents["right_node"], 0.0)
        self.assertEqual(result.currents["forward_node"], 1.0)

    def test_left_and_right_goal_directions_are_signed(self) -> None:
        left = self.encoder.encode(self.position, self.heading, [-5.0, 0.0, -5.0])
        right = self.encoder.encode(self.position, self.heading, [5.0, 0.0, -5.0])
        self.assertLess(left.heading_error_radians, 0.0)
        self.assertGreater(right.heading_error_radians, 0.0)
        self.assertGreater(left.currents["left_node"], left.currents["right_node"])
        self.assertGreater(right.currents["right_node"], right.currents["left_node"])

    def test_right_obstacle_increases_left_avoidance_current(self) -> None:
        result = self.encoder.encode(
            self.position,
            self.heading,
            [0.0, 0.0, -5.0],
            obstacle_positions=[[3.0, 0.0, -2.0]],
        )
        self.assertGreater(result.obstacle_right_cue, 0.0)
        self.assertGreater(result.currents["left_node"], result.currents["right_node"])
        self.assertLess(result.currents["forward_node"], 1.0)

    def test_left_obstacle_increases_right_avoidance_current(self) -> None:
        result = self.encoder.encode(
            self.position,
            self.heading,
            [0.0, 0.0, -5.0],
            obstacle_positions=[[-3.0, 0.0, -2.0]],
        )
        self.assertGreater(result.obstacle_left_cue, 0.0)
        self.assertGreater(result.currents["right_node"], result.currents["left_node"])

    def test_obstacles_outside_sensor_range_do_not_change_currents(self) -> None:
        clear = self.encoder.encode(self.position, self.heading, [0.0, 0.0, -5.0])
        far = self.encoder.encode(
            self.position,
            self.heading,
            [0.0, 0.0, -5.0],
            obstacle_positions=[[20.0, 0.0, -20.0]],
        )
        self.assertEqual(clear.currents, far.currents)

    def test_currents_are_bounded_and_mapping_is_explicit(self) -> None:
        parameters = SensoryEncodingParameters(
            goal_turn_gain=100.0,
            forward_gain=100.0,
            obstacle_avoidance_gain=100.0,
            maximum_current=0.75,
        )
        result = NavigationSensoryEncoder(
            SensoryChannelTargets("L", "R", "F"), parameters
        ).encode(
            self.position,
            self.heading,
            [5.0, 0.0, 0.0],
            obstacle_positions=[[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]],
        )
        self.assertEqual(set(result.currents), {"L", "R", "F"})
        self.assertTrue(all(0.0 <= current <= 0.75 for current in result.currents.values()))

    def test_same_position_goal_has_no_horizontal_drive(self) -> None:
        result = self.encoder.encode(self.position, self.heading, self.position)
        self.assertEqual(result.goal_distance, 0.0)
        self.assertEqual(result.currents["left_node"], 0.0)
        self.assertEqual(result.currents["right_node"], 0.0)
        self.assertEqual(result.currents["forward_node"], 0.0)

    def test_invalid_targets_and_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SensoryChannelTargets("same", "same", "forward")
        with self.assertRaises(ValueError):
            SensoryEncodingParameters(obstacle_forward_suppression=1.1)
        with self.assertRaises(ValueError):
            SensoryEncodingParameters(sensing_radius=0.0)
        with self.assertRaises(ValueError):
            self.encoder.encode(self.position, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0])

    def test_encoding_is_deterministic(self) -> None:
        args = (
            self.position,
            self.heading,
            [2.0, 0.0, -4.0],
            [[-1.0, 0.0, -2.0]],
        )
        first = self.encoder.encode(*args)
        second = self.encoder.encode(*args)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
