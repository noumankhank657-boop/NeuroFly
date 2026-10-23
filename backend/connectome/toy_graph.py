"""Construct the explicitly labelled Phase 4 toy neural graph.

Purpose: provide a tiny reproducible source graph for exercising circuit
propagation before real connectome data is integrated.
Why it exists: the project needs a concrete graph demonstration without
fabricating biological neuron identities or claiming biological provenance.
Inputs: none; parameters are deliberately computational demo settings.
Outputs: a three-node ``ToyNeuralCircuit`` with weighted directed edges.
Dependencies: local LIF and circuit modules.

This graph is **not biological connectome data**. Its node names describe
computational roles only: sensory input, interneuron relay, and motor output.
"""

from __future__ import annotations

from backend.neurons import LIFParameters
from backend.neurons.circuit import ToyNeuralCircuit


def build_demo_circuit() -> ToyNeuralCircuit:
    """Build a three-node one-step-delay excitatory relay circuit."""
    parameters = LIFParameters(
        membrane_time_constant=0.01,
        threshold_potential=1.0,
        reset_potential=0.0,
        time_step=0.01,
    )
    circuit = ToyNeuralCircuit()
    circuit.add_neuron("sensory_input", parameters)
    circuit.add_neuron("interneuron_relay", parameters)
    circuit.add_neuron("motor_output", parameters)
    circuit.add_connection("sensory_input", "interneuron_relay", 2.0)
    circuit.add_connection("interneuron_relay", "motor_output", 2.0)
    return circuit
