"""Tests for the Phase 5 provenance-aware connectome pipeline.

Purpose: verify raw record loading, manifest requirements, filtering, and graph
construction without bundling or fabricating biological data.
Why it exists: provenance and referential integrity are scientific prerequisites
for later neural simulation.
Inputs: temporary CSV/JSON fixtures labelled as external test data.
Outputs: standard-library unittest results.
Dependencies: NetworkX and local connectome modules.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from backend.connectome import GraphFilter, load_connectome_dataset, preprocess_connectome


class ConnectomePipelineTests(unittest.TestCase):
    """Check schema, provenance, filtering, and graph metadata contracts."""

    def _write_dataset(self, root: Path) -> tuple[Path, Path, Path]:
        nodes_path = root / "nodes.csv"
        with nodes_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=["node_id", "name", "cell_type", "region"])
            writer.writeheader()
            writer.writerows(
                [
                    {"node_id": "n1", "name": "left_sensor", "cell_type": "sensory", "region": "FB"},
                    {"node_id": "n2", "name": "relay_a", "cell_type": "interneuron", "region": "FB"},
                    {"node_id": "n3", "name": "relay_b", "cell_type": "interneuron", "region": "PB"},
                    {"node_id": "n4", "name": "motor_a", "cell_type": "motor", "region": "FB"},
                ]
            )
        edges_path = root / "edges.json"
        edges_path.write_text(
            json.dumps(
                [
                    {"source": "n1", "target": "n2", "weight": 2.0},
                    {"source": "n2", "target": "n4", "weight": 0.5},
                    {"source": "n3", "target": "n4", "weight": -1.5},
                ]
            ),
            encoding="utf-8",
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source_name": "External test fixture",
                    "source_type": "external",
                    "source_url": "https://example.invalid/connectome-test-fixture",
                    "version": "test-1",
                    "accessed_at": "2026-10-05",
                    "license": "CC0-1.0",
                    "citation": "Test et al. (2026), synthetic fixture for schema tests",
                    "selected_circuit_purpose": "Verify sensory-relay graph preprocessing",
                    "notes": "Test-only fixture; not a biological claim.",
                }
            ),
            encoding="utf-8",
        )
        return nodes_path, edges_path, manifest_path

    def test_loads_csv_and_json_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_connectome_dataset(*self._write_dataset(Path(directory)))
        self.assertEqual(len(dataset.nodes), 4)
        self.assertEqual(len(dataset.edges), 3)
        self.assertEqual(dataset.edges[2]["weight"], -1.5)
        self.assertEqual(dataset.provenance.source_type, "external")
        self.assertEqual(dataset.provenance.version, "test-1")

    def test_filters_nodes_and_edges_and_attaches_audit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_connectome_dataset(*self._write_dataset(Path(directory)))
        graph, report = preprocess_connectome(
            dataset,
            GraphFilter(cell_types=frozenset({"sensory", "interneuron", "motor"}), minimum_weight=1.0),
        )
        self.assertEqual(set(graph.nodes), {"n1", "n2", "n3", "n4"})
        self.assertEqual(set(graph.edges), {("n1", "n2"), ("n3", "n4")})
        self.assertEqual(report.selected_edge_count, 2)
        self.assertEqual(report.removed_edge_count, 1)
        self.assertEqual(graph.graph["provenance"]["source_name"], "External test fixture")
        self.assertEqual(graph.graph["preprocessing"]["minimum_weight"], 1.0)
        self.assertEqual(graph.graph["preprocessing_report"]["selected_node_count"], 4)

    def test_node_filters_and_deterministic_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_connectome_dataset(*self._write_dataset(Path(directory)))
        graph, _ = preprocess_connectome(
            dataset,
            GraphFilter(regions=frozenset({"FB"}), maximum_nodes=2),
        )
        self.assertEqual(list(graph.nodes), ["n1", "n2"])
        self.assertEqual(list(graph.edges), [("n1", "n2")])

    def test_external_manifest_requires_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_dataset(Path(directory))
            manifest_path = paths[2]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("source_url")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_connectome_dataset(*paths)

    def test_demo_manifest_is_allowed_but_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_dataset(Path(directory))
            manifest_path = paths[2]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_type"] = "demo"
            manifest.pop("source_url")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = load_connectome_dataset(*paths)
        self.assertEqual(dataset.provenance.source_type, "demo")

    def test_unknown_edge_endpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = list(self._write_dataset(Path(directory)))
            paths[1].write_text(json.dumps([{"source": "missing", "target": "n1", "weight": 1}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_connectome_dataset(*paths)

    def test_missing_provenance_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_dataset(Path(directory))
            manifest_path = paths[2]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("citation")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_connectome_dataset(*paths)


if __name__ == "__main__":
    unittest.main()
