"""Control experiments: does the real wiring matter? (Python backend only)

Run from the project root:  python experiments\\control_experiments.py
Compares the real connectome wiring with uniform and shuffled synapse weights.
"""
import random
import sys

sys.path.insert(0, ".")
import numpy as np
from backend.sim.loop import FlySimulation, SimConfig

GOAL = [dict(yaw_deg=y) for y in (-90, -60, -30, 30, 60, 90, 150, -150)] + \
       [dict(start=(x, 7.0), yaw_deg=y) for x, y in ((4, 0), (-4, 0), (5, 40), (-5, -40))]
rng = random.Random(12345)
HELD = []
while len(HELD) < 30:
    o = (rng.uniform(-2.5, 2.5), rng.uniform(-4.5, 4.5), rng.uniform(0.8, 2.0)); yaw = rng.choice([-30, -10, 0, 10, 30])
    try:
        SimConfig(obstacles=[o], yaw_deg=yaw).validate(); HELD.append(dict(obstacles=[o], yaw_deg=yaw))
    except ValueError:
        pass


def tally_fn(cases, fn):
    t = {"reached":0,"collision":0,"timeout":0}
    for k in cases: t[fn(k).status] += 1
    return t

def wired(mutate):
    def fn(k):
        s = FlySimulation(SimConfig(mode="connectome", timeout_s=40, **k)); mutate(s); s.start()
        while s.status == "running": s.step()
        return s
    return fn

def shuffled(seed):
    def m(s):
        edges = list(s.circuit.graph.edges(data=True)); rng = random.Random(seed)
        for sign in (1,-1):
            grp = [a for _,_,a in edges if (a["weight"]>0)==(sign>0)]
            w = [a["weight"] for a in grp]; rng.shuffle(w)
            for a,v in zip(grp,w): a["weight"]=v
    return m

def uniform(s):
    edges = [a for _,_,a in s.circuit.graph.edges(data=True)]
    pos = np.mean([a["weight"] for a in edges if a["weight"]>0]); neg = np.mean([a["weight"] for a in edges if a["weight"]<0])
    for a in edges: a["weight"] = pos if a["weight"]>0 else neg

print("Real connectome wiring:")
print("  goal-only 12:", tally_fn(GOAL, wired(lambda s:None))); print("  30 cases   :", tally_fn(HELD, wired(lambda s:None)))
print("Same wiring but every synapse weight replaced by the average (uniform):")
print("  goal-only 12:", tally_fn(GOAL, wired(uniform))); print("  30 cases   :", tally_fn(HELD, wired(uniform)))
print("Same wiring with weights shuffled between synapses (5 shuffles), 30 cases reached:")
r=[tally_fn(HELD, wired(shuffled(sd)))["reached"] for sd in range(5)]; print("  ",r)
print("  goal-only 12 reached:",[tally_fn(GOAL, wired(shuffled(sd)))["reached"] for sd in range(5)])
