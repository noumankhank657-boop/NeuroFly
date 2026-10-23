"""Connectome-derived DNa02 steering circuit (MaleCNS v1.0 subgraph).

WHAT : a small LIF circuit whose WIRING comes from the male fruit-fly CNS connectome
       (data/connectome/dna02/): 104 presynaptic neurons -> the two DNa02 descending neurons.
WHY  : DNa02 activity predicts steering; turning velocity is linearly related to the
       right-minus-left DNa02 activity (Rayshubskiy et al., eLife). PFL3 neurons compare
       heading with a goal direction and drive DNa02 (Westeinde et al. / Mussells Pires et al.,
       Nature 2024). Left output = DNa02_L, right output = DNa02_R.

WHAT IS DATA                : which neurons connect, how many synapses, which side, the
                              predicted neurotransmitter (all from the dataset export).
WHAT IS OUR ASSUMPTION      : LIF dynamics; sign rule (ACh +, GABA/Glu -); one global
                              scale from synapse count to weight; which input neurons receive
                              the simulated sensory currents (goal drive -> PFL3, obstacle
                              drive -> other excitatory inputs of the side that must turn);
                              and the forward-speed channel, which is a SYNTHETIC neuron here
                              (DNa02 does not encode forward speed).
Nothing here claims to reproduce a real fly's behaviour or brain.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.neurons import LIFParameters
from backend.neurons.circuit import CircuitStepResult, ToyNeuralCircuit
from backend.navigation import SteeringOutputNodes

from .adapter import NavigationCircuitSelection, adapt_connectome_to_lif
from .loader import load_connectome_dataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "connectome" / "dna02"
FORWARD_NODE = "model_forward"   # SYNTHETIC, not from the connectome

PROVENANCE = {
    "kind": "CONNECTOME-DERIVED (MaleCNS v1.0, DNa02 subgraph)",
    "biological": True,
    "note": ("Wiring from male-cns:v1.0; LIF dynamics, sign rule, weight scale and sensory "
             "input mapping are modelling assumptions; forward speed channel is synthetic."),
}


@dataclass(frozen=True)
class Dna02Parameters:
    """Free modelling parameters (assumptions). Tuned in experiments, not measured."""
    edge_scale: float = 0.002          # model weight per synapse
    goal_gain: float = 1.0             # current into PFL3 per unit goal drive
    obstacle_gain: float = 1.0         # current into other excitatory inputs per unit obstacle cue
    tau: float = 0.05
    resistance: float = 8.0


class Dna02SteeringCircuit(ToyNeuralCircuit):
    """ToyNeuralCircuit plus the documented sensory-to-neuron mapping."""

    def __init__(self) -> None:
        super().__init__()
        self.params = Dna02Parameters()
        self.output_nodes: SteeringOutputNodes | None = None
        self.goal_nodes = {"left": [], "right": []}
        self.obstacle_nodes = {"left": [], "right": []}
        self.display_groups: dict[str, list[str]] = {}
        self.disabled: set[str] = set()

    def step_encoded(self, enc) -> CircuitStepResult:
        """Map the encoder's separate goal/obstacle cues onto the selected input neurons."""
        p = self.params
        ext: dict[str, float] = {}
        for n in self.goal_nodes["left"]:
            ext[n] = enc.goal_left_drive * p.goal_gain
        for n in self.goal_nodes["right"]:
            ext[n] = enc.goal_right_drive * p.goal_gain
        # obstacle on the RIGHT -> must turn LEFT -> drive the left-turn side (matches encoder)
        for n in self.obstacle_nodes["left"]:
            ext[n] = enc.obstacle_right_cue * p.obstacle_gain
        for n in self.obstacle_nodes["right"]:
            ext[n] = enc.obstacle_left_cue * p.obstacle_gain
        ext[FORWARD_NODE] = float(enc.currents["sens_forward"])
        for n in self.disabled:           # ablation hook: a disabled neuron gets no input
            ext.pop(n, None)
        return self.step(ext)


def build_dna02_circuit(time_step: float = 0.01, params: Dna02Parameters | None = None,
                        data_dir: Path | str | None = None) -> Dna02SteeringCircuit:
    params = params or Dna02Parameters()
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    dataset = load_connectome_dataset(d / "nodes.csv", d / "edges.csv", d / "manifest.json")
    lif = LIFParameters(membrane_time_constant=params.tau, threshold_potential=1.0,
                        reset_potential=0.0, time_step=time_step, membrane_resistance=params.resistance)
    adapted = adapt_connectome_to_lif(
        dataset,
        NavigationCircuitSelection(name="DNa02 steering subgraph (MaleCNS v1.0)",
                                   purpose=dataset.provenance.selected_circuit_purpose),
        lif)

    c = Dna02SteeringCircuit()
    c.params = params
    c.neurons, c.graph = adapted.circuit.neurons, adapted.circuit.graph
    for _, _, a in c.graph.edges(data=True):
        a["weight"] = float(a["weight"]) * params.edge_scale     # signed synapse count -> weight

    out = {}
    for node, a in c.graph.nodes(data=True):
        if a.get("role") == "output":
            out[a["name"][-1]] = node                            # "L" / "R"
        elif a.get("role") == "input":
            side = "left" if a["target_side"] == "L" else "right"
            if a["cell_type"] == "PFL3":
                c.goal_nodes[side].append(node)
            elif a["nt"] == "acetylcholine":
                c.obstacle_nodes[side].append(node)
    c.add_neuron(FORWARD_NODE, LIFParameters(membrane_time_constant=0.05, threshold_potential=1.0,
                                             reset_potential=0.0, time_step=time_step,
                                             membrane_resistance=8.0))
    c.graph.nodes[FORWARD_NODE].update(role="synthetic-forward", name=FORWARD_NODE)
    c.output_nodes = SteeringOutputNodes.from_names(out["L"], out["R"], FORWARD_NODE)

    c.display_groups = {
        "DNa02_L (out)": [out["L"]], "DNa02_R (out)": [out["R"]],
        "PFL3->L (goal)": c.goal_nodes["left"], "PFL3->R (goal)": c.goal_nodes["right"],
        "otherExc->L": c.obstacle_nodes["left"], "otherExc->R": c.obstacle_nodes["right"],
        "forward (synthetic)": [FORWARD_NODE],
    }
    return c
