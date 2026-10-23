"""NueroFly scaffold runner.

Purpose: validate that the project skeleton, navigation math, LIF neuron, toy
neural graph, connectome pipeline, graph-to-LIF adapter, sensory encoder,
steering decoder, and separate motor model are importable and runnable.
Why it exists: later phases will replace this report with the scientific loop.
Inputs: experiments/configs/default.json and the Phase 2–9 backend packages.
Outputs: a concise architecture/configuration report to stdout.
Dependencies: Python standard library and NumPy through the navigation package.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.connectome import GraphFilter, NavigationCircuitSelection
from backend.connectome.toy_graph import build_demo_circuit
from backend.motor import MotorModel
from backend.navigation import (
    NavigationSensoryEncoder,
    SensoryChannelTargets,
    SteeringDecoder,
    SteeringOutputNodes,
    heading_vector_from_yaw,
    normalize_vector,
)
from backend.neurons import LIFNeuron


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "default.json"
ARCHITECTURE = (
    "DATA",
    "CONNECTOME GRAPH",
    "LIF CIRCUIT",
    "NAVIGATION DECODER",
    "MOTOR MODEL",
    "3D FLY",
    "ENVIRONMENT/SENSORS",
    "BRAIN",
)


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    """Load the JSON configuration used to make the scaffold reproducible."""
    with path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def main() -> int:
    """Report the verified NueroFly boundary without running later simulation code."""
    config = load_config()
    heading = normalize_vector(heading_vector_from_yaw(0.0))
    neuron = LIFNeuron()
    neuron_result = neuron.step(0.0)
    circuit = build_demo_circuit()
    circuit_first = circuit.step({"sensory_input": 2.0})
    circuit_second = circuit.step()
    pipeline_filter = GraphFilter()
    selection = NavigationCircuitSelection(
        name="phase6-template",
        purpose="Require researcher-documented navigation circuit selection",
    )
    sensory = NavigationSensoryEncoder(
        SensoryChannelTargets("left", "right", "forward")
    )
    sensory_result = sensory.encode(
        [0.0, 0.0, 0.0],
        heading_vector_from_yaw(0.0),
        [0.0, 0.0, -1.0],
    )

    decoder_circuit = build_demo_circuit()
    decoder_circuit.add_neuron("left_output")
    decoder_circuit.add_neuron("right_output")
    decoder_circuit.add_neuron("forward_output")
    decoder_circuit.step({"left_output": 2.0, "forward_output": 1.0})
    steering = SteeringDecoder(
        SteeringOutputNodes.from_names("left_output", "right_output", "forward_output")
    ).decode(decoder_circuit)
    motor = MotorModel()
    motor_state = motor.step(steering)

    print(f"NueroFly — Phase {config['phase']} navigation scaffold")
    print("Architecture: " + " -> ".join(ARCHITECTURE))
    print(f"Seed: {config['default_seed']}")
    print(f"Scientific timestep: {config['simulation_timestep_seconds']} s")
    print(f"Reference heading norm: {float((heading ** 2).sum() ** 0.5):.1f}")
    print(
        f"Reference LIF potential: "
        f"{neuron_result.membrane_potential:.1f}; "
        f"spike: {neuron_result.spike}"
    )
    print(
        "Toy graph propagation: "
        f"sensory={circuit_first.spikes['sensory_input']}, "
        f"relay={circuit_second.spikes['interneuron_relay']}"
    )
    print(
        "Connectome pipeline: provenance-required loader; "
        f"default filter max_nodes={pipeline_filter.maximum_nodes}"
    )
    print(
        "LIF adapter policy: "
        f"selection={selection.name}; demo opt-in={selection.allow_demo_data}"
    )
    print(f"Sensory currents: {sensory_result.currents}")
    print(f"Steering command: {steering.as_dict()}")
    print(
        "Motor state: "
        f"position={motor_state.position}, yaw={motor_state.yaw_radians:.4f}, "
        f"forward_velocity={motor_state.forward_velocity:.4f}"
    )
    print(f"Controller training enabled: {config['controller_training']}")
    print("Status: NueroFly navigation math, LIF neuron, toy graph, connectome pipeline, adapter, sensory encoding, decoder, and motor verified; later phases are not implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
