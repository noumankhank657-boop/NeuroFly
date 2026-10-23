"""Compare the neural and connectome controllers on identical scenarios (Python backend only).

Run from the project root:  python experiments\\compare_controllers.py
No tuning is done here: the connectome mode uses its default (first-guess) parameters.
"""
import random
import sys

sys.path.insert(0, ".")
from backend.sim.loop import FlySimulation, SimConfig  # noqa: E402

MODES = ("neural", "connectome")
NAMED = {"no obstacle": [], "obstacle centre": [(0, 0, 1.5)], "obstacle left of line": [(-0.8, 0, 1.5)],
         "obstacle right of line": [(0.8, 0, 1.5)], "obstacle near start": [(0, 4.5, 1.0)],
         "obstacle near goal": [(0, -4.5, 1.0)]}


def run(mode, **kw):
    s = FlySimulation(SimConfig(mode=mode, timeout_s=40, **kw)); s.start()
    while s.status == "running":
        s.step()
    return s


def held_out(n=30, seed=12345):
    rng, cases = random.Random(seed), []
    while len(cases) < n:
        o = (rng.uniform(-2.5, 2.5), rng.uniform(-4.5, 4.5), rng.uniform(0.8, 2.0)); yaw = rng.choice([-30, -10, 0, 10, 30])
        try:
            SimConfig(obstacles=[o], yaw_deg=yaw).validate(); cases.append(dict(obstacles=[o], yaw_deg=yaw))
        except ValueError:
            pass
    return cases


def main():
    print("Named scenarios (status, final distance to destination):")
    for name, o in NAMED.items():
        cells = []
        for m in MODES:
            s = run(m, obstacles=o); cells.append(f"{m}: {s.status} d={s.distance():.1f}")
        print(f"  {name:24s}" + " | ".join(cells))
    print("\n30 random obstacle cases (seed 12345):")
    cases = held_out()
    for m in MODES:
        t = {"reached": 0, "collision": 0, "timeout": 0}
        for k in cases:
            t[run(m, **k).status] += 1
        print(f"  {m:11s}{t}")


if __name__ == "__main__":
    main()
