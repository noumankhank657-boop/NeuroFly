# Digital Fly

**Digital Fly is a 3D navigation simulation with an experimental connectome-derived neural circuit and a reinforcement-learning navigation controller.** It is a research prototype: the main demo runs a tabular Q-learning controller in a planar arena. It is not a complete fly brain, a biological implementation of reinforcement learning, or a model of real flight.

## Project overview

The prototype tests how a simulated agent can move from a start point to a destination while using local obstacle cues. The demo separates three things:

- **Connectome-derived connectivity:** a small MaleCNS v1.0 DNa02 circuit supplies a research wiring diagram. Its LIF dynamics, sensor-to-neuron mapping, synaptic signs and weights are modelling choices.
- **Reinforcement learning:** the main demo uses tabular Q-learning. Its state/action table is an engineering controller, not a biological fly circuit.
- **Simulation input and movement:** Python computes goal bearing and local obstacle cues from exact arena geometry, then drives a planar motor model. Godot renders the state; it does not decide the movement.

The FYP prototype focuses on a reproducible, inspectable navigation demonstration and on keeping the learning and connectome experiments clearly distinguished.

## Architecture

- **Godot 4.7:** builds the 3D arena and controls in `godot/main.gd`, displays destination, trajectory, status and telemetry, and sends commands over TCP.
- **Python 3.10+:** `backend/sim/loop.py` runs the arena, sensors, controller and motor. `backend/sim/server.py` hosts it on `127.0.0.1:8765` by default.
- **Communication:** newline-delimited JSON over a loopback TCP socket. Godot sends `configure`, `start`, `pause`, `resume`, `reset` and `clear_memory` commands. Python sends `nuerofly.state.v2` snapshots at 30 Hz.

## Controller modes

The Godot controller dropdown exposes four modes. It opens on the connectome-derived DNa02 demonstration; Q-learning is the second featured mode. Each run reports its active controller and mode-specific telemetry where applicable.

Interactive trials now have a 180-second simulation-time limit (three minutes), up from 60 seconds. A timeout still reports failure; the longer window gives slow controllers more time but does not imply that a trial is progressing or will reach its goal. The seeded RL evaluation below keeps its explicitly configured 18-second episode limit.

| Dropdown mode | Backend name | What it does |
|---|---|---|
| Connectome-derived DNa02 (research model) | `connectome` | LIF circuit wired from the selected MaleCNS DNa02 subgraph; sensory mapping and forward channel use modelling assumptions. |
| Tabular Q-learning (experimental) | `learning` | Adaptive table controller described below; an engineering controller, not a biological fly circuit. |
| A* waypoint planner | `planned` | Conventional global route planning with waypoints. |
| Synthetic LIF circuit | `neural` | Hand-wired synthetic LIF navigation circuit, not connectome data. |

The older planner and synthetic LIF controllers retain their existing implementations. Their inclusion in the selector does not change their research or engineering limitations.

## Q-learning controller (second focus)

The controller has the following design:

- **State:** 8 quantized goal-bearing sectors, 4 local obstacle-cue categories, a near-obstacle bit and a stuck bit (up to 128 states).
- **Actions:** forward, forward-left, forward-right, pivot-left and pivot-right.
- **Reward:** progress toward the destination earns positive reward; each step has a small cost; blocked movement and repeated unproductive state/action choices are penalized. Reaching the goal earns a terminal reward; timeout, stuck and repeated collision failure receive terminal penalties.
- **Update:** tabular Q-learning, `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) − Q(s,a)]`, with α=0.25 and γ=0.9. Epsilon-greedy exploration uses ε=0.12. A near-goal steering override helps settle the motor at the destination.
- **Learning evidence:** Q-values and the set of visited states change with experience, and selected actions can differ between episodes. The seeded evaluation below does **not** show a consistent improvement in navigation outcomes, so this is an implemented experimental controller rather than a proven improving or reliable agent.
- **Memory:** the JSON Q-table is stored in `data/navigation_memory.json`, loaded on backend startup, saved at episode completion and retained by ordinary Reset. **Clear learned memory + reset** removes the file and returns the simulation to its start state. The saved memory is local runtime data and is git-ignored.

The agent receives simulator-computed goal and obstacle cues, not camera images. The arena is planar, obstacles are circles, and the motor model is deliberately simplified.

## Connectome research component

The `connectome` mode is the default in the main demo selector. It loads a selected subgraph from [data/connectome/dna02](data/connectome/dna02/): 2 DNa02 output neurons, 104 selected presynaptic neurons and 104 edges. `selection_report.json` records that these edges cover 39.6% of the listed DNa02 input synapses.

The source is the **MaleCNS neuPrint dataset `male-cns:v1.0`**, released June 8, 2026. The project uses the derived files `nodes.csv`, `edges.csv` and `manifest.json`; the subgraph was built from the checked-in exports under `data/connectome_export/` by `experiments/build_dna02_subgraph.py`. Dataset license: **CC BY 4.0**. Attribute the source to the FlyEM Project at HHMI Janelia Research Campus, the Cambridge Drosophila Connectomics Group / MRC Laboratory of Molecular Biology, and Google Research; the checked-in data is a selected and transformed subgraph. The official [MaleCNS download page](https://male-cns.janelia.org/download/) provides the dataset files and license. Publication: Berg S, Beckett IR, Costa M, et al. “Sexual dimorphism in the complete Drosophila male central nervous system connectome.” *Cell*. 2026 Sep 3;189(18):5504–5526.e15. doi:[10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015).

This is not a complete connectome simulation. The small DNa02 group is labelled “Prelim Roughly traced” in the checked-in export. The circuit assigns LIF dynamics and a synapse-count-to-weight scale; ACh is treated as excitatory and GABA/glutamate as inhibitory; goal and obstacle cues are mapped to selected cells by hand; and forward speed is supplied by a synthetic neuron. Obstacle-avoidance behavior is therefore a simulator result under these assumptions, not evidence of biological obstacle avoidance.

To run the connectome mode from Python for an experiment:

```python
from backend.sim.loop import FlySimulation, SimConfig

sim = FlySimulation(SimConfig(mode="connectome", obstacles=[(0.0, 0.0, 1.5)]))
sim.start()
for _ in range(6000):
    sim.step()
    if sim.status in ("reached", "timeout", "collision"):
        break
print(sim.status, sim.distance())
```

## Setup and run (Windows)

Install Python 3.10+ and Godot 4.7. From PowerShell in this repository:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_nuerofly.ps1 -GodotPath "C:\path\to\Godot_v4.7-stable_win64.exe"
```

The launcher creates `.venv` if needed, installs `requirements.txt`, starts the Python backend, opens Godot, and stops the backend when Godot exits. First-time dependency installation requires network access. If Godot is already on `PATH`, omit `-GodotPath`. This source project does not include a self-contained Windows build.

## RL training, policy lifecycle, tests and evaluation

Select **Tabular Q-learning** and press **Start** to train through live episodes. The demonstration policy lives at `data/navigation_memory.json`: the backend loads it on startup and saves it after each completed episode. **Reset** preserves that policy; **Clear learned memory + reset** explicitly erases it. There is no separate manual save/load button: completed-episode persistence and startup loading are automatic.

Evaluation uses temporary, isolated tables and cannot alter the demonstration policy:

```powershell
$env:PYTHONPATH = $PWD
.\.venv\Scripts\python.exe -m experiments.evaluate_learning
```

The evaluation reports actual episode statuses, distance, blocked moves and path length. Existing measured outcomes below are mixed and do not establish consistent improvement.

Run the Python suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Verified 2026-10-23 after the prototype changes:** `132 passed`. The suite covers navigation, collision gating, goal tolerance, controller separation, connectome data loading, Q-learning persistence, pause/resume, reset, clear-memory, timeout and stuck reporting. Godot was not installed in this environment, so the visual application and launcher were not manually verified here.

This experiment uses an isolated temporary Q-table; it does not clear the live demo memory. Each layout and seed has four sequential episodes (seeds 2, 11 and 29; timeout 18 simulation seconds per episode):

| Layout | Successes | Failures | Blocked moves |
|---|---:|---:|---:|
| Open arena | 12/12 (100%) | 0 | 0 |
| One centered obstacle | 10/12 (83%) | 2 | 0 |
| Three ordinary obstacles | 5/12 (42%) | 7 | 442 |

Failures include time-limit and stuck states. Across repeated episodes, the open and multi-obstacle cases did not improve consistently; the multi-obstacle success count fell from 2/3 on the first episode to 1/3 on the fourth. The one-obstacle results varied by seed. These small tests show that experience changes the stored Q-table and action choices, but do not establish learning improvement or general reliability. The default demo arena is its own course and these results should not be generalized to arbitrary obstacle layouts.

## Demonstration sequence (2–3 minutes)

1. **Introduce (20–30 s):** identify the 3D Godot visualization, Python simulation/controller, planar movement and tabular Q-learning as separate components.
2. **Run (40–60 s):** show the start and destination, learning controller label, seed, distance, goal tolerance, live trajectory and status. Start the default obstacle layout and let it run until it reaches or reports failure; narrate the actual state shown.
3. **Controls and memory (30–40 s):** pause, show the unchanged step/time, resume, then Reset. Point out that Reset retains learned memory and the episode/state counters.
4. **Learning limit and research context (30–40 s):** optionally clear memory and show the reset. Explain that repeatable tests found mixed results, then describe the DNa02 component as a separately accessible experimental connectome-wired circuit with synthetic sensory and forward-speed assumptions.
5. **Close (10–20 s):** state the main limitations: exact simulator geometry as input, planar arena, circular obstacles, simplified motor dynamics, and no evidence that either controller reproduces biological flight.

## Launch and recording steps

1. Open PowerShell in the repository and run the launcher command above with the installed Godot executable path.
2. Wait for **Backend connected** in the Godot UI. If it does not connect, stop the launcher and check the backend error and whether port 8765 is already in use.
3. Set a repeatable integer seed (default `1`), leave the default destination and obstacle layout, and start the screen recording only after the backend is connected.
4. Follow the demonstration sequence. Keep the final on-screen result honest: **Destination reached** is only shown when measured distance is at or below goal tolerance; otherwise show the reported failure.
5. Stop the recording before closing Godot; the launcher then stops the backend.

Capture genuine screenshots from the running app: (1) the complete arena before Start with visible start, destination and obstacles; (2) an active run with the live trajectory and telemetry; (3) the final goal-reached screen with distance and tolerance visible, or a real failure state if the run fails; and (4) the memory-cleared/reset state if documenting the learning control. Do not fabricate or edit a success state into a screenshot.

## Preserved controller work

Synthetic LIF and A* waypoint implementations remain available in the main controller dropdown. The direct-rule controller has been removed from the app and backend. No global planner is used in the learning controller.

## Known limitations

- RL outcome quality is seed- and layout-dependent; multi-obstacle reliability is poor in the measured scenarios.
- No Godot GUI run or manual screen interaction was possible in the verification environment; launch, rendering and actual Godot-to-Python interaction remain unverified here.
- The TCP server is local-only, and the app has no browser build.
- The environment is deterministic and planar with circular obstacles and simulator-provided geometric sensing.
- Connectome mode uses a small selected MaleCNS subgraph and explicit modelling assumptions; it is not a complete brain or flight model.
- The launcher still requires separate Python and Godot installations.

This is a versioned FYP prototype milestone for future work, not a production navigation system or the end of development.
