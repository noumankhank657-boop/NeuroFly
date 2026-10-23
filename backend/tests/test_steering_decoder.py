"""Tests for Phase 8 interpretable LIF-activity steering decoding.

Purpose: verify continuous left/right/forward commands from circuit activity.
Why it exists: decoder behavior must be explicit and independent of motor
movement before it enters the embodiment loop.
Inputs: small deterministic LIF circuits and output-node mappings.
Outputs: standard-library unittest results.
Dependencies: local neuron and navigation modules.
"""

from __future__ import annotations

import unittest

from backend.navigation import (
    SteeringDecoder,
    SteeringDecoderParameters,
    SteeringOutputNodes,
)
from backend.neurons import LIFParameters, ToyNeuralCircuit


class SteeringDecoderTests(unittest.TestCase):
    """Check activity normalization, continuous output, and safe validation."""

    def _circuit(self) -> ToyNeuralCircuit:
        parameters = LIFParameters(
            membrane_time_constant=0.1,
            threshold_potential=1.0,
            reset_potential=0.0,
            time_step=0.01,
        )
        circuit = ToyNeuralCircuit()
        circuit.add_neuron("left", parameters)
        circuit.add_neuron("right", parameters)
        circuit.add_neuron("forward", parameters)
        circuit.add_neuron("left_2", parameters)
        return circuit

    def test_decode_requires_a_completed_circuit_step(self) -> None:
        decoder = SteeringDecoder(SteeringOutputNodes.from_names("left", "right", "forward"))
        with self.assertRaises(ValueError):
            decoder.decode(self._circuit())

    def test_subthreshold_potential_becomes_continuous_signal(self) -> None:
        circuit = self._circuit()
        circuit.step({"left": 5.0, "right": 0.0, "forward": 2.0})
        command = SteeringDecoder(
            SteeringOutputNodes.from_names("left", "right", "forward"),
            SteeringDecoderParameters(spike_weight=0.0, subthreshold_weight=1.0),
        ).decode(circuit)
        self.assertGreater(command.left, 0.0)
        self.assertGreater(command.forward, 0.0)
        self.assertEqual(command.right, 0.0)
        self.assertEqual(command.step_index, 1)
        self.assertEqual(command.as_dict()["left"], command.left)

    def test_spike_and_subthreshold_activity_are_combined(self) -> None:
        circuit = self._circuit()
        circuit.step({"left": 20.0, "right": 0.0, "forward": 0.0})
        command = SteeringDecoder(
            SteeringOutputNodes.from_names("left", "right", "forward"),
            SteeringDecoderParameters(spike_weight=0.7, subthreshold_weight=0.3),
        ).decode(circuit)
        self.assertTrue(circuit.neurons["left"].spike)
        self.assertAlmostEqual(command.left, 0.7)
        self.assertEqual(command.right, 0.0)
        self.assertEqual(command.forward, 0.0)

    def test_population_activity_is_averaged(self) -> None:
        circuit = self._circuit()
        circuit.step({"left": 5.0, "left_2": 2.0})
        decoder = SteeringDecoder(
            SteeringOutputNodes(("left", "left_2"), ("right",), ("forward",)),
            SteeringDecoderParameters(spike_weight=0.0, subthreshold_weight=1.0),
        )
        command = decoder.decode(circuit)
        expected_left = ((0.5) + (0.2)) / 2.0
        self.assertAlmostEqual(command.raw_left_activity, expected_left)
        self.assertAlmostEqual(command.left, expected_left)

    def test_channel_gains_and_maximum_signal_are_applied(self) -> None:
        circuit = self._circuit()
        circuit.step({"left": 20.0, "right": 20.0, "forward": 20.0})
        decoder = SteeringDecoder(
            SteeringOutputNodes.from_names("left", "right", "forward"),
            SteeringDecoderParameters(
                spike_weight=1.0,
                subthreshold_weight=0.0,
                left_gain=2.0,
                right_gain=0.5,
                forward_gain=3.0,
                maximum_signal=0.8,
            ),
        )
        command = decoder.decode(circuit)
        self.assertEqual(command.left, 0.8)
        self.assertEqual(command.right, 0.5)
        self.assertEqual(command.forward, 0.8)
        self.assertTrue(all(0.0 <= value <= 0.8 for value in command.as_dict().values()))

    def test_unknown_output_nodes_are_rejected(self) -> None:
        circuit = self._circuit()
        circuit.step()
        decoder = SteeringDecoder(SteeringOutputNodes.from_names("missing", "right", "forward"))
        with self.assertRaises(KeyError):
            decoder.decode(circuit)

    def test_mapping_and_parameters_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            SteeringOutputNodes((), ("right",), ("forward",))
        with self.assertRaises(ValueError):
            SteeringOutputNodes(("same",), ("same",), ("forward",))
        with self.assertRaises(ValueError):
            SteeringDecoderParameters(spike_weight=0.0, subthreshold_weight=0.0)
        with self.assertRaises(ValueError):
            SteeringDecoderParameters(maximum_signal=0.0)

    def test_repeated_decode_is_deterministic(self) -> None:
        circuit = self._circuit()
        circuit.step({"left": 3.0, "forward": 1.0})
        decoder = SteeringDecoder(SteeringOutputNodes.from_names("left", "right", "forward"))
        self.assertEqual(decoder.decode(circuit), decoder.decode(circuit))


if __name__ == "__main__":
    unittest.main()
