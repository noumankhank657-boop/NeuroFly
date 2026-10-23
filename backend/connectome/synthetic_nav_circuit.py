"""SYNTHETIC navigation circuit (NOT connectome data).

Purpose: a small hand-wired 9-neuron LIF circuit so the closed loop can run
before real connectome data is supplied.
Why: the supplied ZIP contains no connectome records (see experiments/configs/phase6_selection.json,
status "template-requires-researcher-supplied-connectome"). Inventing biological
IDs would be dishonest, so every neuron here is named by computational role only.
Structure: sensory(L,R,F) -> relay(L,R,F) -> output(L,R,F), plus mutual
left/right inhibition at the relay stage (a simple winner-take-all-ish steering
contrast). Weights are hand-chosen, not measured.
"""
from __future__ import annotations

from backend.neurons import LIFParameters
from backend.neurons.circuit import ToyNeuralCircuit
from backend.navigation import SensoryChannelTargets, SteeringOutputNodes

PROVENANCE = {
    "kind": "SYNTHETIC",
    "biological": False,
    "note": "Hand-wired demo circuit; weights are not derived from any connectome.",
}

SENSORY_TARGETS = SensoryChannelTargets("sens_left", "sens_right", "sens_forward")
OUTPUT_NODES = SteeringOutputNodes.from_names("out_left", "out_right", "out_forward")


def build_synthetic_nav_circuit(time_step: float = 0.01) -> ToyNeuralCircuit:
    # NueroFly fix: tau 0.02->0.05 s and R 4->8. With the old values any input above
    # ~0.5 made a neuron fire on EVERY step, so left/right currents of 0.73 vs 0.68
    # produced identical (saturated) outputs and the steering difference was lost.
    # With these values the firing rate stays graded over the encoder's 0-2 range.
    # Hand-chosen, found by sweep (see NUEROFLY_FIX_NOTES.md); not measured biology.
    p = LIFParameters(membrane_time_constant=0.05, threshold_potential=1.0,
                      reset_potential=0.0, time_step=time_step,
                      membrane_resistance=8.0)
    c = ToyNeuralCircuit()
    for n in ("sens_left", "sens_right", "sens_forward",
              "relay_left", "relay_right", "relay_forward",
              "out_left", "out_right", "out_forward"):
        c.add_neuron(n, p)
    for side in ("left", "right", "forward"):
        c.add_connection(f"sens_{side}", f"relay_{side}", 1.5)
        c.add_connection(f"relay_{side}", f"out_{side}", 1.5)
    c.add_connection("relay_left", "relay_right", -1.0)   # mutual inhibition
    c.add_connection("relay_right", "relay_left", -1.0)
    return c
