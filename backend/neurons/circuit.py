"""Small synchronous neural circuit built from LIF neurons and weighted edges.

Purpose: propagate activity through a transparent toy graph before introducing
real connectome-derived data.
Why it exists: Phase 4 demonstrates how individual LIF neurons become a circuit
without hiding computation inside a black-box controller.
Inputs: named LIF neurons, directed weighted connections, and external currents.
Outputs: per-step input currents, membrane potentials, and spike states.
Dependencies: NetworkX and the local LIF implementation.

Timing rule
-----------
The circuit is synchronous. At step ``t``, every edge reads the source spike
state produced at step ``t-1`` and contributes ``weight * previous_spike`` to the
target's current. All neurons then advance once. This creates an explicit one
simulation-step propagation delay and avoids dependence on dictionary/graph
iteration order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import networkx as nx

from .lif import LIFNeuron, LIFParameters


@dataclass(frozen=True, slots=True)
class CircuitStepResult:
    """Immutable observation of one synchronous circuit update."""

    step_index: int
    input_currents: dict[str, float]
    membrane_potentials: dict[str, float]
    spikes: dict[str, bool]


class ToyNeuralCircuit:
    """A named directed graph whose nodes are independently testable LIF neurons."""

    def __init__(self) -> None:
        """Create an empty directed circuit; nodes and edges are added explicitly."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.neurons: dict[str, LIFNeuron] = {}
        self.step_index = 0
        self.last_result: CircuitStepResult | None = None

    def add_neuron(
        self,
        name: str,
        parameters: LIFParameters | None = None,
        *,
        initial_potential: float | None = None,
    ) -> LIFNeuron:
        """Add one uniquely named LIF node and return its neuron instance."""
        self._validate_name(name)
        if name in self.neurons:
            raise ValueError(f"neuron already exists: {name}")
        neuron = LIFNeuron(parameters, initial_potential=initial_potential)
        self.neurons[name] = neuron
        self.graph.add_node(name)
        return neuron

    def add_connection(self, source: str, target: str, weight: float) -> None:
        """Add or replace a directed weighted synaptic connection.

        Positive weights are excitatory in this computational model and negative
        weights are inhibitory. Replacing an edge is explicit and deterministic;
        parallel biological synapses are outside this toy graph's scope.
        """
        self._require_neuron(source)
        self._require_neuron(target)
        if not math.isfinite(weight):
            raise ValueError("connection weight must be finite")
        self.graph.add_edge(source, target, weight=float(weight))

    def step(self, external_currents: Mapping[str, float] | None = None) -> CircuitStepResult:
        """Advance all nodes once using the previous step's spike snapshot.

        WHAT: sum external currents and weighted previous spikes, then step every
        LIF neuron exactly once.
        WHY: synchronous updates make causal timing explicit and reproducible.
        OUTPUT: currents delivered, post-step potentials, and current spike flags.
        """
        external = dict(external_currents or {})
        unknown = set(external) - set(self.neurons)
        if unknown:
            raise KeyError(f"external current names are not neurons: {sorted(unknown)}")
        for name, current in external.items():
            if not math.isfinite(current):
                raise ValueError(f"external current for {name} must be finite")

        currents = {name: float(external.get(name, 0.0)) for name in self.neurons}
        previous_spikes = {name: neuron.spike for name, neuron in self.neurons.items()}
        for source, target, attributes in self.graph.edges(data=True):
            if previous_spikes[source]:
                currents[target] += float(attributes["weight"])

        potentials: dict[str, float] = {}
        spikes: dict[str, bool] = {}
        for name, neuron in self.neurons.items():
            result = neuron.step(currents[name])
            potentials[name] = result.membrane_potential
            spikes[name] = result.spike

        self.step_index += 1
        self.last_result = CircuitStepResult(
            step_index=self.step_index,
            input_currents=currents,
            membrane_potentials=potentials,
            spikes=spikes,
        )
        return self.last_result

    def reset(self) -> None:
        """Reset every neuron and the circuit clock for a new deterministic trial."""
        for neuron in self.neurons.values():
            neuron.reset()
        self.step_index = 0
        self.last_result = None

    def _require_neuron(self, name: str) -> None:
        """Raise a clear error when an edge references an absent node."""
        self._validate_name(name)
        if name not in self.neurons:
            raise KeyError(f"unknown neuron: {name}")

    @staticmethod
    def _validate_name(name: str) -> None:
        """Validate node identifiers before they enter graph state."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("neuron name must be a non-empty string")
