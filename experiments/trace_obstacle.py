import math, sys
sys.path.insert(0, ".")
from backend.sim.loop import FlySimulation, SimConfig
import numpy as np
from backend.navigation import heading_vector_from_yaw

def run(obs, label, steps=6000, every=150):
    cfg = SimConfig(obstacles=[obs])
    s = FlySimulation(cfg); s.start()
    print("==", label, obs)
    print("step  x      z     yaw_deg  speed  | sensL  sensR  fwd | cmdL  cmdR  cmdF | obsL obsR | status")
    for i in range(steps):
        pos = np.array(s.motor.position); goal = np.array([cfg.destination[0],0,cfg.destination[1]])
        h = heading_vector_from_yaw(s.motor.yaw_radians)
        enc = s.encoder.encode(pos, h, goal, s.arena.sensed_points(pos[0],pos[2]))
        s.step()
        if i % every == 0 or s.status != "running":
            c = enc.currents
            print(f"{i:5d} {s.motor.position[0]:6.2f} {s.motor.position[2]:6.2f} {math.degrees(s.motor.yaw_radians):7.1f} {s.motor.forward_velocity:5.2f} | "
                  f"{c['sens_left']:.3f} {c['sens_right']:.3f} {c['sens_forward']:.3f} | "
                  f"{s.last_cmd['left']:.3f} {s.last_cmd['right']:.3f} {s.last_cmd['forward']:.3f} | "
                  f"{enc.obstacle_left_cue:.2f} {enc.obstacle_right_cue:.2f} | {s.status}")
        if s.status != "running": break
    print("final:", s.status, s.message, "dist", round(s.distance(),2), "steps", s.steps)

run((0.0, 0.0, 1.5), "centered obstacle")

