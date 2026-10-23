"""Tests for the connectome-derived DNa02 steering circuit (MaleCNS v1.0 subgraph).

These check wiring, sign handling, labelling and closed-loop behaviour in the PYTHON backend.
They do not prove anything about a real fly, and they do not test Godot.
"""
from backend.connectome.dna02_circuit import (FORWARD_NODE, build_dna02_circuit)
from backend.connectome.loader import load_connectome_dataset
from backend.connectome.dna02_circuit import DEFAULT_DATA_DIR
from backend.sim.loop import FlySimulation, SimConfig

DNA02_L, DNA02_R = "523769", "10360"


def _run(mode="connectome", seconds=40, mutate=None, **kw):
    s = FlySimulation(SimConfig(mode=mode, timeout_s=seconds, **kw))
    if mutate:
        mutate(s)
    s.start()
    while s.status == "running":
        s.step()
    return s


def test_provenance_is_external_and_versioned():
    d = DEFAULT_DATA_DIR
    ds = load_connectome_dataset(d / "nodes.csv", d / "edges.csv", d / "manifest.json")
    assert ds.provenance.source_type == "external"
    assert "male-cns:v1.0" in ds.provenance.version
    assert ds.provenance.source_url


def test_circuit_structure_matches_export():
    c = build_dna02_circuit()
    assert len(c.neurons) == 107               # 104 inputs + 2 DNa02 + 1 synthetic forward neuron
    assert c.output_nodes.left == (DNA02_L,) and c.output_nodes.right == (DNA02_R,)
    assert len(c.goal_nodes["left"]) == 12 and len(c.goal_nodes["right"]) == 12   # PFL3
    assert c.graph.graph["provenance"]["source_type"] == "external"


def test_edge_signs_follow_neurotransmitter_rule():
    c = build_dna02_circuit()
    for u, v, a in c.graph.edges(data=True):
        nt = c.graph.nodes[u]["nt"]
        if nt == "acetylcholine":
            assert a["weight"] > 0
        elif nt in ("gaba", "glutamate"):
            assert a["weight"] < 0


def test_forward_channel_is_labelled_synthetic():
    c = build_dna02_circuit()
    assert c.output_nodes.forward == (FORWARD_NODE,)
    assert c.graph.nodes[FORWARD_NODE]["role"] == "synthetic-forward"


def test_left_goal_drives_left_dna02_and_mirror():
    for yaw, winner, loser in ((60, DNA02_L, DNA02_R), (-60, DNA02_R, DNA02_L)):
        s = FlySimulation(SimConfig(mode="connectome", yaw_deg=yaw)); s.start()
        n = {DNA02_L: 0, DNA02_R: 0}
        for _ in range(60):
            s.step()
            for k in n:
                n[k] += int(s.last_spikes[k])
        assert n[winner] > n[loser], (yaw, n)


def test_goal_navigation_off_axis_uses_the_wiring():
    for yaw in (-90, 60, 150):
        s = _run(yaw_deg=yaw)
        assert s.status == "reached", (yaw, s.status, s.distance())


def test_control_removing_pfl3_input_breaks_goal_steering():
    def ablate(s):
        s.circuit.disabled = set(s.circuit.goal_nodes["left"] + s.circuit.goal_nodes["right"])
    s = _run(yaw_deg=60, seconds=20, mutate=ablate)
    assert s.status != "reached"


def test_control_zero_synaptic_weights_break_steering():
    def zero(s):
        for _, _, a in s.circuit.graph.edges(data=True):
            a["weight"] = 0.0
    s = _run(yaw_deg=60, seconds=20, mutate=zero)
    assert s.status != "reached"


def test_control_swapped_left_right_routing_breaks_steering():
    def swap(s):
        g = s.circuit.goal_nodes
        g["left"], g["right"] = g["right"], g["left"]
    s = _run(yaw_deg=60, seconds=20, mutate=swap)
    assert s.status != "reached"


def test_centered_obstacle_is_cleared_by_connectome_mode():
    s = _run(obstacles=[(0.0, 0.0, 1.5)])
    assert s.status == "reached", (s.status, s.distance())


def test_snapshots_are_labelled_and_modes_stay_separate():
    a = FlySimulation(SimConfig(mode="connectome")).snapshot(0)
    b = FlySimulation(SimConfig(mode="neural")).snapshot(0)
    assert "CONNECTOME-DERIVED" in a["neural"]["provenance"] and a["neural"].get("grouped")
    assert b["neural"]["provenance"] == "SYNTHETIC" and not b["neural"].get("grouped")
    assert "synthetic" in a["controller_note"].lower()   # forward channel disclosed
