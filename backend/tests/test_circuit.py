"""Unit tests for the Phase 4 toy neural graph.

Purpose: verify NetworkX graph structure and causal LIF spike propagation.
Why it exists: graph timing and edge weights must be correct before a biological
adapter or navigation circuit is introduced.
Inputs: small deterministic computational graph fixtures.
Outputs: standard-library unittest results.
Dependencies: NetworkX and local neuron modules.
"""

from __future__ import annotations

import unittest

from backend.connectome.toy_graph import build_demo_circuit
from backend.neurons import LIFParameters, ToyNeuralCircuit


class ToyNeuralCircuitTests(unittest.TestCase):
    """Check explicit graph structure and synchronous one-step causality."""

    def _chain(self) -> ToyNeuralCircuit:
        parameters = LIFParameters(
            membrane_time_constant=0.01,
            threshold_potential=1.0,
            time_step=0.01,
        )
        circuit = ToyNeuralCircuit()
        circuit.add_neuron("source", parameters)
        circuit.add_neuron("target", parameters)
        circuit.add_connection("source", "target", 2.0)
        return circuit

    def test_nodes_and_weighted_edges_are_explicit(self) -> None:
        circuit = self._chain()
        self.assertEqual(set(circuit.graph.nodes), {"source", "target"})
        self.assertEqual(circuit.graph["source"]["target"]["weight"], 2.0)

    def test_spikes_propagate_after_one_step_delay(self) -> None:
        circuit = self._chain()
        first = circuit.step({"source": 2.0})
        self.assertTrue(first.spikes["source"])
        self.assertFalse(first.spikes["target"])
        self.assertEqual(first.input_currents["target"], 0.0)

        second = circuit.step({"source": 0.0})
        self.assertFalse(second.spikes["source"])
        self.assertTrue(second.spikes["target"])
        self.assertEqual(second.input_currents["target"], 2.0)

    def test_negative_weight_is_inhibitory_current(self) -> None:
        circuit = self._chain()
        circuit.add_connection("source", "target", -0.5)
        first = circuit.step({"source": 2.0})
        self.assertTrue(first.spikes["source"])
        second = circuit.step({"source": 0.0})
        self.assertEqual(second.input_currents["target"], -0.5)
        self.assertLess(second.membrane_potentials["target"], 0.0)
        self.assertFalse(second.spikes["target"])

    def test_reset_clears_neurons_clock_and_last_result(self) -> None:
        circuit = self._chain()
        circuit.step({"source": 2.0})
        circuit.reset()
        self.assertEqual(circuit.step_index, 0)
        self.assertIsNone(circuit.last_result)
        result = circuit.step()
        self.assertEqual(result.step_index, 1)
        self.assertFalse(any(result.spikes.values()))

    def test_external_current_and_edge_names_are_validated(self) -> None:
        circuit = self._chain()
        with self.assertRaises(KeyError):
            circuit.step({"missing": 1.0})
        with self.assertRaises(KeyError):
            circuit.add_connection("source", "missing", 1.0)
        with self.assertRaises(ValueError):
            circuit.add_neuron("")
        with self.assertRaises(ValueError):
            circuit.add_connection("source", "target", float("nan"))

    def test_demo_graph_is_three_node_and_propagates(self) -> None:
        circuit = build_demo_circuit()
        self.assertEqual(len(circuit.graph.nodes), 3)
        self.assertEqual(len(circuit.graph.edges), 2)
        first = circuit.step({"sensory_input": 2.0})
        second = circuit.step()
        third = circuit.step()
        self.assertTrue(first.spikes["sensory_input"])
        self.assertTrue(second.spikes["interneuron_relay"])
        self.assertTrue(third.spikes["motor_output"])

    def test_same_inputs_produce_same_spike_trace(self) -> None:
        def trace() -> list[dict[str, bool]]:
            circuit = build_demo_circuit()
            return [circuit.step({"sensory_input": 2.0}).spikes,
                    circuit.step().spikes,
                    circuit.step().spikes]

        self.assertEqual(trace(), trace())


if __name__ == "__main__":
    unittest.main()
