"""Provenance-aware loading and validation for connectome node/edge data.

Purpose: convert documented CSV/JSON node and connection records into a typed
in-memory dataset suitable for graph preprocessing.
Why it exists: biological provenance and schema validity must be checked before
any data can influence a neural simulation.
Inputs: node records, edge records, and a JSON provenance manifest.
Outputs: ``ConnectomeDataset`` containing normalized records and a validated
``ProvenanceManifest``.
Dependencies: Python standard library only.

No biological dataset is bundled by this module. A manifest may identify an
external dataset or an explicitly labelled demo source. The loader never fills
missing biological fields with invented values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any


NODE_COLUMNS = frozenset({"node_id", "name", "cell_type"})
EDGE_COLUMNS = frozenset({"source", "target", "weight"})
MANIFEST_FIELDS = frozenset(
    {
        "source_name",
        "source_type",
        "version",
        "accessed_at",
        "license",
        "citation",
        "selected_circuit_purpose",
    }
)


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """Required provenance record attached to every loaded dataset."""

    source_name: str
    source_type: str
    version: str
    accessed_at: str
    license: str
    citation: str
    selected_circuit_purpose: str
    source_url: str | None = None
    notes: str | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ProvenanceManifest":
        """Validate and construct a manifest without inferring missing facts."""
        missing = sorted(field for field in MANIFEST_FIELDS if not str(values.get(field, "")).strip())
        if missing:
            raise ValueError(f"provenance manifest missing required fields: {missing}")
        source_type = str(values["source_type"]).strip().lower()
        if source_type not in {"external", "demo"}:
            raise ValueError("source_type must be 'external' or 'demo'")
        source_url_value = values.get("source_url")
        source_url = None if source_url_value is None else str(source_url_value).strip()
        if source_type == "external" and not source_url:
            raise ValueError("external provenance requires source_url")
        notes_value = values.get("notes")
        notes = None if notes_value is None else str(notes_value).strip()
        return cls(
            source_name=str(values["source_name"]).strip(),
            source_type=source_type,
            version=str(values["version"]).strip(),
            accessed_at=str(values["accessed_at"]).strip(),
            license=str(values["license"]).strip(),
            citation=str(values["citation"]).strip(),
            selected_circuit_purpose=str(values["selected_circuit_purpose"]).strip(),
            source_url=source_url or None,
            notes=notes or None,
        )

    def as_dict(self) -> dict[str, str]:
        """Return a serializable provenance snapshot for graph metadata."""
        result = {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "version": self.version,
            "accessed_at": self.accessed_at,
            "license": self.license,
            "citation": self.citation,
            "selected_circuit_purpose": self.selected_circuit_purpose,
        }
        if self.source_url:
            result["source_url"] = self.source_url
        if self.notes:
            result["notes"] = self.notes
        return result


@dataclass(frozen=True, slots=True)
class ConnectomeDataset:
    """Normalized records plus provenance, before filtering into a graph."""

    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    provenance: ProvenanceManifest


def load_connectome_dataset(
    nodes_path: str | Path,
    edges_path: str | Path,
    manifest_path: str | Path,
) -> ConnectomeDataset:
    """Load, normalize, and validate a connectome dataset from CSV or JSON.

    WHAT: parse node/edge records, coerce identifiers and weights, validate
    required fields and referential integrity, then attach the manifest.
    WHY: malformed or undocumented biological data must fail before graph
    construction rather than silently changing scientific results.
    OUTPUT: immutable ``ConnectomeDataset`` with normalized dictionaries.
    """
    manifest_values = _read_json_object(Path(manifest_path))
    provenance = ProvenanceManifest.from_mapping(manifest_values)
    raw_nodes = _read_records(Path(nodes_path), expected_key="nodes")
    raw_edges = _read_records(Path(edges_path), expected_key="edges")
    nodes = tuple(_normalize_node(record, index) for index, record in enumerate(raw_nodes))
    edges = tuple(_normalize_edge(record, index) for index, record in enumerate(raw_edges))
    node_ids = [record["node_id"] for record in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("node_id values must be unique")
    known_ids = set(node_ids)
    missing_endpoints = sorted(
        {
            endpoint
            for edge in edges
            for endpoint in (edge["source"], edge["target"])
            if endpoint not in known_ids
        }
    )
    if missing_endpoints:
        raise ValueError(f"edges reference unknown node_id values: {missing_endpoints}")
    return ConnectomeDataset(nodes=nodes, edges=edges, provenance=provenance)


def _read_records(path: Path, *, expected_key: str) -> list[dict[str, Any]]:
    """Read a list of dictionaries from CSV or a list/object JSON payload."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as input_file:
            return [dict(row) for row in csv.DictReader(input_file)]
    if path.suffix.lower() == ".json":
        payload = _read_json_object(path)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload.get(expected_key), list):
            records = payload[expected_key]
        else:
            raise ValueError(f"{path} must contain a list or a '{expected_key}' list")
        if not all(isinstance(record, dict) for record in records):
            raise ValueError(f"{path} records must be JSON objects")
        return [dict(record) for record in records]
    raise ValueError(f"unsupported data format for {path}; use .csv or .json")


def _read_json_object(path: Path) -> dict[str, Any] | list[Any]:
    """Read JSON while preserving the distinction between object and list payloads."""
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"{path} must contain a JSON object or list")
    return payload


def _normalize_node(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Validate required node fields and preserve documented optional attributes."""
    _require_columns(record, NODE_COLUMNS, f"node record {index}")
    node_id = str(record["node_id"]).strip()
    name = str(record["name"]).strip()
    cell_type = str(record["cell_type"]).strip()
    if not node_id or not name or not cell_type:
        raise ValueError(f"node record {index} has an empty required field")
    normalized = dict(record)
    normalized.update({"node_id": node_id, "name": name, "cell_type": cell_type})
    return normalized


def _normalize_edge(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Validate edge endpoints and convert the connection weight to a finite float."""
    _require_columns(record, EDGE_COLUMNS, f"edge record {index}")
    source = str(record["source"]).strip()
    target = str(record["target"]).strip()
    if not source or not target:
        raise ValueError(f"edge record {index} has an empty endpoint")
    try:
        weight = float(record["weight"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"edge record {index} has a non-numeric weight") from error
    if not math.isfinite(weight):
        raise ValueError(f"edge record {index} weight must be finite")
    normalized = dict(record)
    normalized.update({"source": source, "target": target, "weight": weight})
    return normalized


def _require_columns(record: Mapping[str, Any], required: Iterable[str], label: str) -> None:
    """Raise a compact schema error for missing required record fields."""
    missing = sorted(column for column in required if column not in record)
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")
