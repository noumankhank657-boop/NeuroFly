"""Unit tests for the Phase 3 Leaky Integrate-and-Fire neuron.

Purpose: verify the explicit membrane equation and state transitions.
Why it exists: spike behavior must be reproducible and inspectable before graph
connectivity is introduced.
Inputs: deterministic LIF parameter fixtures and scalar currents.
Outputs: standard-library unittest results.
Dependencies: Python standard library and the project neuron package.
"""

from __future__ import annotations

import unittest

from backend.neurons import LIFNeuron, LIFParameters


class LIFNeuronTests(unittest.TestCase):
    """Check the computational contract of a single LIF neuron."""

    def test_default_state_is_quiet_at_rest(self) -> None:
        neuron = LIFNeuron()
        result = neuron.step(0.0)
        self.assertFalse(result.spike)
        self.assertAlmostEqual(result.membrane_potential, 0.0)
        self.assertEqual(neuron.spike_count, 0)

    def test_leak_returns_potential_toward_rest(self) -> None:
        parameters = LIFParameters(
            membrane_time_constant=0.1,
            resting_potential=0.0,
            reset_potential=0.0,
            threshold_potential=10.0,
            time_step=0.01,
        )
        neuron = LIFNeuron(parameters, initial_potential=1.0)
        result = neuron.step(0.0)
        self.assertAlmostEqual(result.membrane_potential, 0.9)
        self.assertFalse(result.spike)

    def test_input_current_integrates_subthreshold(self) -> None:
        parameters = LIFParameters(
            membrane_time_constant=0.1,
            threshold_potential=1.0,
            time_step=0.01,
        )
        neuron = LIFNeuron(parameters)
        result = neuron.step(5.0)
        self.assertAlmostEqual(result.membrane_potential, 0.5)
        self.assertEqual(result.input_current, 5.0)
        self.assertFalse(result.spike)

    def test_threshold_crossing_emits_spike_and_resets(self) -> None:
        parameters = LIFParameters(
            membrane_time_constant=0.02,
            threshold_potential=1.0,
            reset_potential=-0.25,
            time_step=0.01,
        )
        neuron = LIFNeuron(parameters)
        result = neuron.step(2.5)
        self.assertTrue(result.spike)
        self.assertAlmostEqual(result.membrane_potential, -0.25)
        self.assertEqual(neuron.spike_count, 1)
        self.assertAlmostEqual(neuron.membrane_potential, parameters.reset_potential)

    def test_spike_flag_is_transient_on_next_quiet_step(self) -> None:
        parameters = LIFParameters(
            membrane_time_constant=0.01,
            threshold_potential=1.0,
            time_step=0.01,
        )
        neuron = LIFNeuron(parameters)
        self.assertTrue(neuron.step(2.0).spike)
        self.assertFalse(neuron.step(0.0).spike)

    def test_refractory_period_suppresses_immediate_spikes(self) -> None:
        parameters = LIFParameters(
            membrane_time_constant=0.01,
            threshold_potential=1.0,
            time_step=0.01,
            refractory_period=0.02,
        )
        neuron = LIFNeuron(parameters)
        self.assertTrue(neuron.step(2.0).spike)
        self.assertFalse(neuron.step(2.0).spike)
        self.assertFalse(neuron.step(2.0).spike)
        self.assertTrue(neuron.step(2.0).spike)
        self.assertEqual(neuron.spike_count, 2)

    def test_reset_clears_dynamic_state(self) -> None:
        parameters = LIFParameters(
            membrane_time_constant=0.01,
            threshold_potential=1.0,
            time_step=0.01,
        )
        neuron = LIFNeuron(parameters)
        neuron.step(2.0)
        neuron.reset()
        self.assertEqual(neuron.spike_count, 0)
        self.assertFalse(neuron.spike)
        self.assertEqual(neuron.input_current, 0.0)
        self.assertEqual(neuron.membrane_potential, parameters.reset_potential)

    def test_invalid_parameters_and_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LIFParameters(membrane_time_constant=0.0)
        with self.assertRaises(ValueError):
            LIFParameters(threshold_potential=0.0, reset_potential=1.0)
        with self.assertRaises(ValueError):
            LIFNeuron(initial_potential=float("nan"))
        with self.assertRaises(ValueError):
            LIFNeuron().step(float("inf"))
        with self.assertRaises(ValueError):
            LIFNeuron().step(0.0, time_step=0.0)


if __name__ == "__main__":
    unittest.main()
