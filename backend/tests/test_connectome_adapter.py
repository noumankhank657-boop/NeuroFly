"""Tests for Phase 6 selected-circuit adaptation to LIF neurons.

Purpose: verify that a validated selected graph becomes an inspectable LIF
circuit without losing provenance or inventing biological metadata.
Why it exists: this adapter is the boundary between data engineering and neural
simulation, so source policy and one-to-one mapping must be tested.
Inputs: temporary synthetic fixtures with explicit demo or external labels.
Outputs: standard-library unittest results.
Dependencies: NetworkX and local connectome modules.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from backend.connectome import (
    GraphFilter,
    NavigationCircuitSelection,
    adapt_connectome_to_lif,
    load_connectome_dataset,
)
from backend.neurons import LIFParameters


class ConnectomeAdapterTests(unittest.TestCase):
    """Check source gating, one-to-one mapping, metadata, and propagation."""

    def _dataset(self, root: Path, source_type: str = "demo"):
        nodes_path = root / "nodes.csv"
        with nodes_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(writer_file := output, fieldnames=["node_id", "name", "cell_type", "region"])
            writer.writeheader()
            writer.writerows(
                [
                    {"node_id": "sensor", "name": "sensor_node", "cell_type": "sensory", "region": "navigation"},
                    {"node_id": "relay", "name": "relay_node", "cell_type": "interneuron", "region": "navigation"},
                    {"node_id": "other", "name": "other_node", "cell_type": "unselected", "region": "other"},
                ]
            )
        edges_path = root / "edges.json"
        edges_path.write_text(
            json.dumps(
                [
                    {"source": "sensor", "target": "relay", "weight": 2.0},
                    {"source": "relay", "target": "other", "weight": 1.0},
                ]
            ),
            encoding="utf-8",
        )
        manifest_path = root / "manifest.json"
        manifest = {
            "source_name": "Phase 6 test fixture",
            "source_type": source_type,
            "version": "fixture-1",
            "accessed_at": "2026-10-05",
            "license": "CC0-1.0",
            "citation": "Temporary test fixture; not biological evidence",
            "selected_circuit_purpose": "Test selected circuit conversion",
        }
        if source_type == "external":
            manifest["source_url"] = "https://example.invalid/phase6-fixture"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return load_connectome_dataset(nodes_path, edges_path, manifest_path)

    def _selection(self, allow_demo_data: bool = False) -> NavigationCircuitSelection:
        return NavigationCircuitSelection(
            name="test-navigation-circuit",
            purpose="Test conversion from selected graph to LIF circuit",
            graph_filter=GraphFilter(regions=frozenset({"navigation"})),
            allow_demo_data=allow_demo_data,
        )

    def test_demo_data_is_blocked_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), source_type="demo")
        with self.assertRaises(ValueError):
            adapt_connectome_to_lif(dataset, self._selection())

    def test_explicit_demo_opt_in_converts_selected_nodes_and_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), source_type="demo")
        adapted = adapt_connectome_to_lif(dataset, self._selection(allow_demo_data=True))
        self.assertEqual(adapted.source_type, "demo")
        self.assertEqual(adapted.node_count, 2)
        self.assertEqual(adapted.edge_count, 1)
        self.assertEqual(set(adapted.circuit.neurons), {"sensor", "relay"})
        self.assertEqual(adapted.circuit.graph["sensor"]["relay"]["weight"], 2.0)
        self.assertEqual(adapted.circuit.graph.nodes["sensor"]["cell_type"], "sensory")
        self.assertEqual(adapted.circuit.graph.graph["adapter"], "connectome-to-lif")
        self.assertEqual(adapted.circuit.graph.graph["selection"]["name"], "test-navigation-circuit")

    def test_external_data_can_convert_and_propagate_spikes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), source_type="external")
        parameters = LIFParameters(
            membrane_time_constant=0.01,
            threshold_potential=1.0,
            time_step=0.01,
        )
        adapted = adapt_connectome_to_lif(dataset, self._selection(), parameters)
        first = adapted.circuit.step({"sensor": 2.0})
        second = adapted.circuit.step()
        self.assertTrue(first.spikes["sensor"])
        self.assertFalse(first.spikes["relay"])
        self.assertTrue(second.spikes["relay"])
        self.assertEqual(adapted.preprocessing_report.selected_node_count, 2)

    def test_empty_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), source_type="external")
        selection = NavigationCircuitSelection(
            name="empty",
            purpose="Expected empty selection test",
            graph_filter=GraphFilter(node_ids=frozenset({"missing"})),
        )
        with self.assertRaises(ValueError):
            adapt_connectome_to_lif(dataset, selection)

    def test_selection_requires_name_and_purpose(self) -> None:
        with self.assertRaises(ValueError):
            NavigationCircuitSelection(name="", purpose="purpose")
        with self.assertRaises(ValueError):
            NavigationCircuitSelection(name="name", purpose="")


if __name__ == "__main__":
    unittest.main()
