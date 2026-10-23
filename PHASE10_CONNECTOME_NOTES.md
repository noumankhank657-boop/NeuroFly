# Connectome-derived DNa02 steering circuit (MaleCNS v1.0)

Mode name in the simulator: `connectome`. The older `neural` mode is the SYNTHETIC baseline,
`direct` is a plain rule baseline, `planned` is A* waypoint navigation, and `learning` is tabular
Q-learning. They are kept separate and labelled in snapshots. The main Godot demo now exposes only
the experimental learning controller; these earlier modes remain research/legacy entry points.

## Dataset (verified on the official page)
- MaleCNS, adult male Drosophila central nervous system (Janelia FlyEM + Cambridge + MRC LMB).
- neuPrint dataset name `male-cns:v1.0`, server https://neuprint.janelia.org, accessed 2026-10-21.
- License: CC BY 4.0 (Creative Commons Attribution 4.0 International), confirmed on the official
  [MaleCNS download page](https://male-cns.janelia.org/download/) and the paper.
- Citation: Berg S, Beckett IR, Costa M, et al. *Sexual dimorphism in the complete Drosophila male
  central nervous system connectome.* Cell. 2026 Sep 3;189(18):5504-5526.e15.
  doi:10.1016/j.cell.2026.08.015.
- Release: MaleCNS v1.0 was released June 8, 2026; the paper was published September 3, 2026.
- DNa02 itself is labelled "Prelim Roughly traced" in this release; 305 of 1095 inputs are too.

## Why DNa02 / PFL3 (evidence, from papers - not from this data)
- DNa02: recordings show it predicts steering; turning velocity is linearly related to the
  right-minus-left DNa02 activity (Rayshubskiy et al., eLife). DNa01 role judged unclear -> not used.
- PFL3: compares heading with a goal direction and drives steering (Westeinde et al.; Mussells
  Pires et al., Nature 2024). In this export PFL3 gives 736 synapses onto DNa02: 24 neurons, all
  predicted acetylcholine, all CONTRALATERAL (soma side R -> DNa02_L and L -> DNa02_R).
- No evidence was found for an obstacle-avoidance input to DNa02. Obstacle handling is an assumption.

## Subgraph (experiments/build_dna02_subgraph.py -> data/connectome/dna02/)
2 DNa02 neurons + 104 presynaptic neurons (top 40 per side by synapse count + all PFL3) = 39.6% of
DNa02's input synapses. 73 excitatory and 31 inhibitory edges; every input neuron targets one side.

## What is data vs assumption
DATA: neurons, synapse counts, sides, predicted neurotransmitter, DNa02 L/R identity.
ASSUMPTIONS: LIF dynamics (tau 0.05, R 8); ACh=+, GABA/Glu=-; one scale 0.002 weight per synapse;
goal drive -> PFL3 on the side that must turn; obstacle drive -> other excitatory inputs of the
side that must turn; inhibitory inputs receive no sensory current; the forward-speed channel is a
SYNTHETIC neuron (DNa02 does not encode forward speed). The decoder reads DNa02_L/R rates.

## Results (Python backend only; Godot NOT tested)
Control experiments, 12 off-axis goal-only starts: intact 12/12 reached; PFL3 input removed 0/12;
all synaptic weights 0 -> 0/12; left/right PFL3 routing swapped -> 0/12.
Edge scale (0.001-0.004) x obstacle gain (0.5-1.5): identical results on the tuning set (5/6, 12/12).
Defaults were NOT tuned.

| 30 random obstacle cases | reached | collision | timeout |
|---|---|---|---|
| direct (no obstacle sensing) | 9 | 21 | 0 |
| neural (synthetic, retuned) | 27 | 1 | 2 |
| connectome (DNa02) | 25 | 1 | 4 |

Named scenarios: connectome reaches 5/6; fails "obstacle near goal" (timeout, 0.8 from goal).
In the one failing case traced, both DNa02 outputs saturate (left=right=0.70, yaw 0) - the same tie
problem as the old synthetic circuit. The other timeouts end 0.7-1.4 units from the goal; their cause
was NOT traced. Obstacle avoidance here is a model result, not evidence about real flies.

Reproduce: `python experiments\compare_controllers.py`, `python -m pytest -q`.

## Honest limits found by extra controls (python experiments\control_experiments.py)
| controller | goal-only 12 | 30 random obstacle cases (reached) |
|---|---|---|
| plain program, no neurons, senses obstacles | 12/12 | 29/30 |
| real connectome wiring | 12/12 | 25/30 |
| same wiring, every weight set to the average | 12/12 | 25/30 |
| same wiring, weights shuffled (5 shuffles) | 12/12 each | 25/30 each |

- A simple program with no neurons does at least as well as the connectome circuit.
- The exact synapse counts do NOT change the result here. What matters (earlier controls) is the
  structure: which side each neuron feeds, and that the inputs are excitatory.
- The simulator's encoder already works out "turn left/right" (goal_left_drive etc.) before the
  circuit sees it, so the circuit mostly passes a ready-made decision to DNa02. It does not decide.
- Everything is deterministic: no noise. The seed 12345 only picks the 30 test obstacle layouts.
- The circuit is 107 neurons (of ~166,000 in the dataset); input is exact simulator geometry,
  not vision; no memory, no learning. Do not describe it as intelligent or fly-like.
