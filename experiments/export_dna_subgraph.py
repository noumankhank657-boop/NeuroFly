"""Small read-only export of the DNa steering neurons + their inputs from MaleCNS v1.0.

Usage (PowerShell, from the project root):
    $env:NEUPRINT_TOKEN = "<your token>"
    python experiments\\export_dna_subgraph.py

Writes CSV files into data\\connectome_export\\ . Nothing in the simulator is changed.
Dataset: male-cns:v1.0 on https://neuprint.janelia.org (as named on the official download page).
"""
import os
import sys
from pathlib import Path
import pandas as pd
from neuprint import Client, NeuronCriteria as NC, fetch_neurons, fetch_simple_connections

DATASET = "male-cns:v1.0"
SERVER = "https://neuprint.janelia.org"
OUT = Path("data") / "connectome_export"
MIN_WEIGHT = 5          # ignore connections with fewer than 5 synapses (keeps the file small)
TARGET_TYPES = ["DNa02", "DNa01"]


def main():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit("NEUPRINT_TOKEN is not set. See the instructions to get a token.")
    OUT.mkdir(parents=True, exist_ok=True)
    client = Client(SERVER, dataset=DATASET, token=token)
    print("Connected:", SERVER, DATASET)

    # 1) what DNa* types exist in this dataset (so we never guess a name)
    dna = fetch_neurons(NC(type="DNa.*", regex=True), omit_rois=True)  # omit_rois=True -> ONE table
    summary = dna.groupby("type").size().rename("n_neurons").reset_index()
    summary.to_csv(OUT / "dna_types_found.csv", index=False)
    print("\nDNa* types found:\n", summary.to_string(index=False))

    for t in TARGET_TYPES:
        neurons = fetch_neurons(NC(type=t), omit_rois=True)
        if len(neurons) == 0:
            print(f"\n!! No neurons of type {t} found. Check dna_types_found.csv.")
            continue
        neurons.to_csv(OUT / f"{t}_neurons.csv", index=False)
        print(f"\n{t}: {len(neurons)} neurons (bodyIds: {list(neurons.bodyId)})")

        # 2) who feeds this type (inputs), >= MIN_WEIGHT synapses
        inp = fetch_simple_connections(upstream_criteria=None,
                                       downstream_criteria=NC(type=t),
                                       min_weight=MIN_WEIGHT,
                                       properties=["type", "instance"])
        inp.to_csv(OUT / f"{t}_inputs.csv", index=False)
        top = (inp.groupby("type_pre", dropna=False)["weight"].sum()
                  .sort_values(ascending=False).head(15))
        print(f"{t}: {len(inp)} input connections; top input TYPES by total synapses:")
        print(top.to_string())

        # 3) what do the input neurons themselves look like (class, nt prediction, side ...)
        ids = sorted(set(inp.bodyId_pre))
        if ids:
            pre = fetch_neurons(NC(bodyId=ids), omit_rois=True)
            pre.to_csv(OUT / f"{t}_input_neurons.csv", index=False)

    print("\nDone. Files are in", OUT.resolve())


if __name__ == "__main__":
    main()