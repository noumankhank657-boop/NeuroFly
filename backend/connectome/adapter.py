"""Adapt a selected connectome graph into the LIF circuit representation.

Purpose: bridge the Phase 5 selected NetworkX graph to Phase 3/4 LIF neurons.
Why it exists: biological or external graph records must enter neural dynamics
through an explicit, inspectable adapter rather than implicit dictionary logic.
Inputs: a validated ``ConnectomeDataset``, a named circuit selection, and LIF
parameters shared by the converted computational neurons.
Outputs: ``AdaptedCircuit`` containing the selected graph, audit report, and a
``ToyNeuralCircuit`` populated with one LIF neuron per selected node.
Dependencies: NetworkX and local loader, preprocessing, and neuron modules.

The adapter is provenance-gated. External datasets are accepted when their
manifest is complete. Demo data is rejected by default and can only be used when
the caller explicitly sets ``allow_demo_data=True``. No biological identity,
function, synapse type, or parameter is inferred by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from backend.neurons import LIFParameters, ToyNeuralCircuit

from .loader import ConnectomeDataset
from .preprocess import GraphFilter, PreprocessingReport, preprocess_connectome


@dataclass(frozen=True, slots=True)
class NavigationCircuitSelection:
    """Documented selection intent for one small navigation-related subgraph."""

    name: str
    purpose: str
    graph_filter: GraphFilter = GraphFilter()
    allow_demo_data: bool = False

    def __post_init__(self) -> None:
        """Require a human-readable circuit name and computational purpose."""
        if not self.name.strip():
            raise ValueError("selection name must be non-empty")
        if not self.purpose.strip():
            raise ValueError("selection purpose must be non-empty")


@dataclass(frozen=True, slots=True)
class AdaptedCircuit:
    """Selected graph, conversion result, and audit information."""

    circuit: ToyNeuralCircuit
    selected_graph: nx.DiGraph
    selection: NavigationCircuitSelection
    preprocessing_report: PreprocessingReport
    source_type: str

    @property
    def node_count(self) -> int:
        """Return the number of LIF neurons created by the adapter."""
        return len(self.circuit.neurons)

    @property
    def edge_count(self) -> int:
        """Return the number of weighted connections created by the adapter."""
        return self.circuit.graph.number_of_edges()


def adapt_connectome_to_lif(
    dataset: ConnectomeDataset,
    selection: NavigationCircuitSelection,
    neuron_parameters: LIFParameters | None = None,
) -> AdaptedCircuit:
    """Select a documented graph and convert every selected node/edge to LIF.

    WHAT: enforce source policy, preprocess the dataset, create one LIF neuron
    per selected node, copy node metadata, and copy finite edge weights.
    WHY: this is the explicit biological-data-to-computation boundary; it keeps
    selection, provenance, graph structure, and neural state inspectable.
    OUTPUT: an ``AdaptedCircuit`` ready for future sensory input mapping.
    """
    _enforce_source_policy(dataset, selection)
    selected_graph, report = preprocess_connectome(dataset, selection.graph_filter)
    if selected_graph.number_of_nodes() == 0:
        raise ValueError("circuit selection produced no nodes")

    parameters = neuron_parameters or LIFParameters()
    circuit = ToyNeuralCircuit()
    for node_id, attributes in selected_graph.nodes(data=True):
        circuit.add_neuron(node_id, parameters)
        circuit.graph.nodes[node_id].update(attributes)
    for source, target, attributes in selected_graph.edges(data=True):
        circuit.add_connection(source, target, float(attributes["weight"]))

    circuit.graph.graph.update(
        {
            "provenance": dict(selected_graph.graph["provenance"]),
            "preprocessing": dict(selected_graph.graph["preprocessing"]),
            "preprocessing_report": dict(selected_graph.graph["preprocessing_report"]),
            "selection": {
                "name": selection.name,
                "purpose": selection.purpose,
                "allow_demo_data": selection.allow_demo_data,
            },
            "adapter": "connectome-to-lif",
        }
    )
    return AdaptedCircuit(
        circuit=circuit,
        selected_graph=selected_graph,
        selection=selection,
        preprocessing_report=report,
        source_type=dataset.provenance.source_type,
    )


def _enforce_source_policy(
    dataset: ConnectomeDataset,
    selection: NavigationCircuitSelection,
) -> None:
    """Prevent an unlabeled demo graph from being presented as biological data."""
    if dataset.provenance.source_type == "demo" and not selection.allow_demo_data:
        raise ValueError(
            "demo data is blocked by default; set allow_demo_data=True for an explicit demo run"
        )
    if dataset.provenance.source_type not in {"external", "demo"}:
        raise ValueError("unsupported provenance source_type")
