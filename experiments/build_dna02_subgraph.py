"""Build the small DNa02 steering subgraph from the neuPrint export (no network needed).

Input : data/connectome_export/DNa02_inputs.csv, DNa02_input_neurons.csv, DNa02_neurons.csv
        (made by experiments/export_dna_subgraph.py from male-cns:v1.0)
Output: data/connectome/dna02/{nodes.csv, edges.csv, manifest.json, selection_report.json}

Selection rule (documented, deterministic, NOT chosen by guessing names):
  * outputs  = the two DNa02 neurons (L and R)
  * inputs   = for each DNa02 side, the TOP_N presynaptic neurons by synapse count
               PLUS every PFL3 neuron that synapses onto DNa02 (literature: goal steering)
  * edge sign: acetylcholine -> excitatory (+), GABA / glutamate -> inhibitory (-).
               Edges whose transmitter is another/unclear are dropped and counted.
               THIS SIGN RULE IS A MODELLING ASSUMPTION, not read from the data.
  * edge weight in edges.csv = signed synapse count (raw, unscaled)
"""
import csv
import json
from pathlib import Path

import pandas as pd

SRC = Path("data/connectome_export")
OUT = Path("data/connectome/dna02")
TOP_N = 40
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = pd.read_csv(SRC / "DNa02_inputs.csv")
    pre = pd.read_csv(SRC / "DNa02_input_neurons.csv").set_index("bodyId")
    dn = pd.read_csv(SRC / "DNa02_neurons.csv")
    side_of = {int(r.bodyId): r.instance[-1] for r in dn.itertuples()}  # L / R from instance name
    assert sorted(side_of.values()) == ["L", "R"], side_of

    keep = set()
    for body in side_of:
        top = inputs[inputs.bodyId_post == body].sort_values("weight", ascending=False).head(TOP_N)
        keep |= set(zip(top.bodyId_pre, top.bodyId_post))
    pfl3 = inputs[inputs.type_pre == "PFL3"]
    keep |= set(zip(pfl3.bodyId_pre, pfl3.bodyId_post))

    sel = inputs[[(a, b) in keep for a, b in zip(inputs.bodyId_pre, inputs.bodyId_post)]].copy()
    sel["nt"] = sel.bodyId_pre.map(pre["consensusNt"])
    dropped = sel[~sel.nt.isin(SIGN)]
    sel = sel[sel.nt.isin(SIGN)].copy()
    sel["sign"] = sel.nt.map(SIGN)
    sel["signed_weight"] = sel.sign * sel.weight

    nodes = []
    for body, side in side_of.items():
        nodes.append(dict(node_id=str(body), name=f"DNa02_{side}", cell_type="DNa02", role="output",
                          soma_side=side, nt="acetylcholine", target_side=""))
    for body in sorted(set(sel.bodyId_pre)):
        r = pre.loc[body]
        edges_b = sel[sel.bodyId_pre == body]
        strongest = edges_b.sort_values("weight", ascending=False).iloc[0]
        nodes.append(dict(node_id=str(body), name=str(r["instance"]), cell_type=str(r["type"]), role="input",
                          soma_side=str(r.get("somaSide", "")), nt=str(r["consensusNt"]),
                          target_side=side_of[int(strongest.bodyId_post)]))
    pd.DataFrame(nodes).to_csv(OUT / "nodes.csv", index=False)
    pd.DataFrame({"source": sel.bodyId_pre.astype(str), "target": sel.bodyId_post.astype(str),
                  "weight": sel.signed_weight, "synapses": sel.weight, "nt": sel.nt}).to_csv(OUT / "edges.csv", index=False)

    manifest = {
        "source_name": "MaleCNS connectome (Janelia FlyEM + Cambridge + MRC LMB, adult male Drosophila CNS)",
        "source_type": "external",
        "source_url": "https://male-cns.janelia.org",
        "version": "male-cns:v1.0 (neuPrint dataset name, server https://neuprint.janelia.org)",
        "accessed_at": "2026-10-21",
        "license": "CC-BY (as stated on male-cns.janelia.org)",
        "citation": "Janelia FlyEM male CNS v1.0 - exact paper citation to be confirmed from https://male-cns.janelia.org before the report",
        "selected_circuit_purpose": "DNa02 steering readout and its direct presynaptic inputs for goal/obstacle steering",
        "notes": "Edge weight = signed synapse count; sign from consensus neurotransmitter (ACh +, GABA/Glu -), a modelling assumption. DNa02 itself is labelled 'Prelim Roughly traced' in this release.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_in = len(nodes) - 2
    report = {
        "top_n_per_side": TOP_N, "n_input_neurons": n_in, "n_edges": int(len(sel)),
        "n_pfl3_neurons": int((pd.DataFrame(nodes).cell_type == "PFL3").sum()),
        "dropped_edges_other_or_unclear_nt": int(len(dropped)),
        "excitatory_edges": int((sel.sign > 0).sum()), "inhibitory_edges": int((sel.sign < 0).sum()),
        "share_of_all_DNa02_input_synapses": round(float(sel.weight.sum() / inputs.weight.sum()), 3),
    }
    (OUT / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
