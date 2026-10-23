"""Control experiments: does the real wiring matter? (Python backend only)

Run from the project root:  python experiments\\control_experiments.py
Compares: a plain no-neuron program that senses obstacles, the real connectome wiring,
the same wiring with every weight replaced by the average, and with weights shuffled.
"""
import random
import sys

sys.path.insert(0, ".")
import numpy as np
from backend.sim.loop import FlySimulation, SimConfig
from backend.navigation import heading_vector_from_yaw

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


def rule_run(cfg):
    """A plain program, NO neurons: encoder currents used directly as commands (it does sense obstacles)."""
    s = FlySimulation(SimConfig(mode="direct", timeout_s=40, **cfg)); s.start()
    c0 = s.config
    while s.status == "running":
        pos = np.array(s.motor.position); goal = np.array([c0.destination[0],0.0,c0.destination[1]])
        h = heading_vector_from_yaw(s.motor.yaw_radians)
        enc = s.encoder.encode(pos, h, goal, s.arena.sensed_points(pos[0], pos[2]))
        c = enc.currents
        cmd = {"left": min(1.0,c["sens_left"]), "right": min(1.0,c["sens_right"]), "forward": min(1.0,c["sens_forward"])}
        st = s.motor.step(cmd); s.steps += 1; s.sim_time += 0.01
        if s.arena.collision(st.position[0], st.position[2]): s.status = "collision"
        elif s.distance() <= c0.tolerance: s.status = "reached"
        elif s.sim_time >= c0.timeout_s: s.status = "timeout"
    return s

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

print("Plain program WITH obstacle sensing, no neurons:")
print("  goal-only 12:", tally_fn(GOAL, rule_run)); print("  30 cases   :", tally_fn(HELD, rule_run))
print("Real connectome wiring:")
print("  goal-only 12:", tally_fn(GOAL, wired(lambda s:None))); print("  30 cases   :", tally_fn(HELD, wired(lambda s:None)))
print("Same wiring but every synapse weight replaced by the average (uniform):")
print("  goal-only 12:", tally_fn(GOAL, wired(uniform))); print("  30 cases   :", tally_fn(HELD, wired(uniform)))
print("Same wiring with weights shuffled between synapses (5 shuffles), 30 cases reached:")
r=[tally_fn(HELD, wired(shuffled(sd)))["reached"] for sd in range(5)]; print("  ",r)
print("  goal-only 12 reached:",[tally_fn(GOAL, wired(shuffled(sd)))["reached"] for sd in range(5)])
