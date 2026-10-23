"""Filter validated connectome records into a documented NetworkX subgraph.

Purpose: make the raw-to-selected-circuit stage explicit and reproducible.
Why it exists: the neural model should receive a small, justified graph rather
than an unexamined entire dataset.
Inputs: ``ConnectomeDataset`` and a ``GraphFilter`` configuration.
Outputs: a NetworkX directed graph plus a ``PreprocessingReport``.
Dependencies: NetworkX and the local connectome loader.

Filtering is intentionally structural and transparent: node IDs, cell types,
regions, name fragments, edge-weight threshold, and an optional deterministic
node limit. It does not infer biological function or claim that the selected
nodes form a complete biological circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .loader import ConnectomeDataset


@dataclass(frozen=True, slots=True)
class GraphFilter:
    """Explicit criteria for selecting a small graph from loaded records."""

    node_ids: frozenset[str] = frozenset()
    cell_types: frozenset[str] = frozenset()
    regions: frozenset[str] = frozenset()
    name_contains: str | None = None
    minimum_weight: float | None = None
    maximum_nodes: int | None = None

    def __post_init__(self) -> None:
        """Validate filter values so preprocessing cannot silently drift."""
        if self.minimum_weight is not None and self.minimum_weight < 0.0:
            raise ValueError("minimum_weight must be non-negative")
        if self.maximum_nodes is not None and self.maximum_nodes <= 0:
            raise ValueError("maximum_nodes must be positive")
        if self.name_contains is not None and not self.name_contains.strip():
            raise ValueError("name_contains must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class PreprocessingReport:
    """Counts and filter record saved alongside a selected graph."""

    input_node_count: int
    input_edge_count: int
    selected_node_count: int
    selected_edge_count: int
    removed_node_count: int
    removed_edge_count: int
    filters: dict[str, Any] = field(default_factory=dict)


def preprocess_connectome(
    dataset: ConnectomeDataset,
    graph_filter: GraphFilter | None = None,
) -> tuple[nx.DiGraph, PreprocessingReport]:
    """Filter records, construct a directed graph, and preserve provenance.

    WHAT: select nodes by explicit metadata, retain eligible edges, and attach
    provenance/filter/report metadata to the graph.
    WHY: this is the auditable bridge from source records to a small circuit;
    every reduction is measurable and repeatable.
    OUTPUT: ``(graph, report)`` where graph attributes include ``provenance``
    and ``preprocessing`` dictionaries.
    """
    settings = graph_filter or GraphFilter()
    selected_nodes = [
        node for node in dataset.nodes if _node_matches(node, settings)
    ]
    if settings.maximum_nodes is not None:
        selected_nodes = sorted(selected_nodes, key=lambda node: node["node_id"])[
            : settings.maximum_nodes
        ]
    selected_ids = {node["node_id"] for node in selected_nodes}
    selected_edges = [
        edge
        for edge in dataset.edges
        if edge["source"] in selected_ids
        and edge["target"] in selected_ids
        and _edge_matches(edge, settings)
    ]

    graph = nx.DiGraph()
    graph.graph["provenance"] = dataset.provenance.as_dict()
    graph.graph["preprocessing"] = _filter_as_dict(settings)
    for node in selected_nodes:
        node_id = node["node_id"]
        attributes = {key: value for key, value in node.items() if key != "node_id"}
        graph.add_node(node_id, **attributes)
    for edge in selected_edges:
        source = edge["source"]
        target = edge["target"]
        attributes = {key: value for key, value in edge.items() if key not in {"source", "target"}}
        graph.add_edge(source, target, **attributes)

    report = PreprocessingReport(
        input_node_count=len(dataset.nodes),
        input_edge_count=len(dataset.edges),
        selected_node_count=graph.number_of_nodes(),
        selected_edge_count=graph.number_of_edges(),
        removed_node_count=len(dataset.nodes) - graph.number_of_nodes(),
        removed_edge_count=len(dataset.edges) - graph.number_of_edges(),
        filters=_filter_as_dict(settings),
    )
    graph.graph["preprocessing_report"] = {
        "input_node_count": report.input_node_count,
        "input_edge_count": report.input_edge_count,
        "selected_node_count": report.selected_node_count,
        "selected_edge_count": report.selected_edge_count,
        "removed_node_count": report.removed_node_count,
        "removed_edge_count": report.removed_edge_count,
        "filters": report.filters,
    }
    return graph, report


def _node_matches(node: dict[str, Any], settings: GraphFilter) -> bool:
    """Apply all configured node predicates using documented attributes only."""
    if settings.node_ids and node["node_id"] not in settings.node_ids:
        return False
    if settings.cell_types and node["cell_type"] not in settings.cell_types:
        return False
    if settings.regions and str(node.get("region", "")) not in settings.regions:
        return False
    if settings.name_contains and settings.name_contains.lower() not in node["name"].lower():
        return False
    return True


def _edge_matches(edge: dict[str, Any], settings: GraphFilter) -> bool:
    """Apply the optional absolute-weight threshold to connection records."""
    return settings.minimum_weight is None or abs(edge["weight"]) >= settings.minimum_weight


def _filter_as_dict(settings: GraphFilter) -> dict[str, Any]:
    """Serialize filter choices for audit logs and reproducibility manifests."""
    return {
        "node_ids": sorted(settings.node_ids),
        "cell_types": sorted(settings.cell_types),
        "regions": sorted(settings.regions),
        "name_contains": settings.name_contains,
        "minimum_weight": settings.minimum_weight,
        "maximum_nodes": settings.maximum_nodes,
    }
