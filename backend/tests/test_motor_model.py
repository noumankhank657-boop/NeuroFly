"""Tests for the NueroFly separate motor model.

Purpose: verify steering-to-yaw, speed, and position dynamics.
Why it exists: motor state must be correct and independent before a 3D renderer
or collision system consumes it.
Inputs: bounded steering mappings and decoder-compatible commands.
Outputs: standard-library unittest results.
Dependencies: NumPy and local navigation/motor modules.
"""

from __future__ import annotations

import math
import unittest

from backend.motor import MotorModel, MotorParameters
from backend.navigation import SteeringCommand


class MotorModelTests(unittest.TestCase):
    """Check independent motor dynamics and state validation."""

    def setUp(self) -> None:
        self.parameters = MotorParameters(
            time_step=0.1,
            maximum_forward_speed=2.0,
            maximum_yaw_rate=math.pi,
            forward_acceleration=4.0,
            forward_deceleration=6.0,
        )

    def test_zero_command_keeps_initial_state(self) -> None:
        motor = MotorModel(self.parameters)
        state = motor.step({"left": 0.0, "right": 0.0, "forward": 0.0})
        self.assertEqual(state.position, (0.0, 0.0, 0.0))
        self.assertEqual(state.yaw_radians, 0.0)
        self.assertEqual(state.yaw_rate, 0.0)
        self.assertEqual(state.forward_velocity, 0.0)
        self.assertEqual(state.step_index, 1)

    def test_forward_command_accelerates_and_moves_along_negative_z(self) -> None:
        motor = MotorModel(self.parameters)
        first = motor.step({"left": 0.0, "right": 0.0, "forward": 1.0})
        self.assertAlmostEqual(first.forward_velocity, 0.4)
        self.assertAlmostEqual(first.position[0], 0.0)
        self.assertAlmostEqual(first.position[2], -0.04)
        second = motor.step({"left": 0.0, "right": 0.0, "forward": 1.0})
        self.assertAlmostEqual(second.forward_velocity, 0.8)
        self.assertAlmostEqual(second.position[2], -0.12)

    def test_differential_steering_changes_yaw_direction(self) -> None:
        motor = MotorModel(self.parameters)
        right_turn = motor.step({"left": 0.0, "right": 1.0, "forward": 0.0})
        self.assertAlmostEqual(right_turn.yaw_rate, math.pi)
        self.assertAlmostEqual(right_turn.yaw_radians, math.pi * 0.1)
        motor.reset()
        left_turn = motor.step({"left": 1.0, "right": 0.0, "forward": 0.0})
        self.assertAlmostEqual(left_turn.yaw_rate, -math.pi)
        self.assertAlmostEqual(left_turn.yaw_radians, -math.pi * 0.1)

    def test_speed_decelerates_with_separate_limit(self) -> None:
        motor = MotorModel(self.parameters)
        motor.step({"left": 0.0, "right": 0.0, "forward": 1.0})
        motor.step({"left": 0.0, "right": 0.0, "forward": 1.0})
        state = motor.step({"left": 0.0, "right": 0.0, "forward": 0.0})
        self.assertAlmostEqual(state.forward_velocity, 0.2)

    def test_accepts_decoder_command_object(self) -> None:
        motor = MotorModel(self.parameters)
        command = SteeringCommand(
            left=0.0,
            right=0.0,
            forward=0.5,
            raw_left_activity=0.0,
            raw_right_activity=0.0,
            raw_forward_activity=0.5,
            step_index=7,
        )
        state = motor.step(command)
        self.assertGreater(state.forward_velocity, 0.0)

    def test_reset_is_reproducible_and_accepts_new_pose(self) -> None:
        motor = MotorModel(self.parameters, initial_position=(1.0, 2.0, 3.0), initial_yaw_radians=0.5)
        motor.step({"left": 0.0, "right": 1.0, "forward": 1.0})
        state = motor.reset(position=(4.0, 5.0, 6.0), yaw_radians=-0.5)
        self.assertEqual(state.position, (4.0, 5.0, 6.0))
        self.assertAlmostEqual(state.yaw_radians, -0.5)
        self.assertEqual(state.forward_velocity, 0.0)
        self.assertEqual(state.step_index, 0)

    def test_yaw_wraps_for_stable_state(self) -> None:
        motor = MotorModel(self.parameters, initial_yaw_radians=math.pi)
        state = motor.step({"left": 0.0, "right": 1.0, "forward": 0.0})
        self.assertGreaterEqual(state.yaw_radians, -math.pi)
        self.assertLess(state.yaw_radians, math.pi)

    def test_invalid_parameters_positions_and_commands_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MotorParameters(maximum_forward_speed=0.0)
        with self.assertRaises(ValueError):
            MotorModel(initial_position=(0.0, 0.0))
        motor = MotorModel(self.parameters)
        with self.assertRaises(KeyError):
            motor.step({"left": 0.0, "right": 0.0})
        with self.assertRaises(ValueError):
            motor.step({"left": 0.0, "right": 0.0, "forward": 1.1})
        with self.assertRaises(TypeError):
            motor.step("not a command")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            motor.step({"left": 0.0, "right": 0.0, "forward": 0.0}, time_step=0.0)

    def test_same_command_trace_is_deterministic(self) -> None:
        commands = [
            {"left": 0.0, "right": 0.0, "forward": 1.0},
            {"left": 0.0, "right": 0.4, "forward": 0.8},
            {"left": 0.2, "right": 0.0, "forward": 0.3},
        ]

        def trace() -> list[tuple[float, ...]]:
            motor = MotorModel(self.parameters)
            return [
                (
                    state.position[0],
                    state.position[2],
                    state.yaw_radians,
                    state.forward_velocity,
                )
                for command in commands
                for state in [motor.step(command)]
            ]

        self.assertEqual(trace(), trace())


if __name__ == "__main__":
    unittest.main()
