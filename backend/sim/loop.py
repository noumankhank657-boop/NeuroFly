"""The closed brain-to-body loop.

state -> sensory currents -> LIF circuit -> steering decoder -> motor model -> state

Three controllers, kept strictly separate and labelled in every snapshot:
  "neural"     : SYNTHETIC hand-wired LIF circuit (baseline, not biology)
  "connectome" : LIF circuit whose wiring is the MaleCNS v1.0 DNa02 subgraph
                 (forward speed channel is synthetic; see dna02_circuit.py)
  "direct"     : a plain heading-error rule, used ONLY as a comparison baseline
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

import numpy as np

from backend.connectome.synthetic_nav_circuit import (
    OUTPUT_NODES, PROVENANCE, SENSORY_TARGETS, build_synthetic_nav_circuit)
from backend.connectome.dna02_circuit import (PROVENANCE as DNA02_PROVENANCE,
                                              build_dna02_circuit)
from backend.motor import MotorModel
from backend.navigation import (NavigationSensoryEncoder, SteeringDecoder,
                                SteeringDecoderParameters, heading_vector_from_yaw,
                                signed_heading_error)
from backend.navigation.planner import plan_route
from .environment import Arena, Obstacle

SCHEMA = "nuerofly.state.v2"
DT = 0.01  # seconds per simulation step
RATE_WINDOW_STEPS = 10  # decoder averages the last 10 steps (0.1 s) into a firing-rate estimate


@dataclass
class SimConfig:
    start: tuple[float, float] = (0.0, 7.0)      # (x, z)
    yaw_deg: float = 0.0                         # 0 = facing -Z, + = right
    destination: tuple[float, float] = (0.0, -7.0)
    mode: str = "neural"                         # planned | neural | connectome | direct
    tolerance: float = 0.5                       # arrival radius
    safety_margin: float = 0.15                  # clearance beyond fly/obstacle radii
    timeout_s: float = 60.0
    obstacles: list[tuple[float, float, float]] = field(default_factory=list)

    def validate(self) -> None:
        vals = [*self.start, self.yaw_deg, *self.destination, self.tolerance, self.timeout_s, self.safety_margin]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError("all coordinates must be finite numbers")
        if self.mode not in ("planned", "neural", "connectome", "direct"):
            raise ValueError("mode must be 'planned', 'neural', 'connectome' or 'direct'")
        if self.tolerance <= 0 or self.timeout_s <= 0 or self.safety_margin < 0:
            raise ValueError("tolerance and timeout must be positive and safety_margin non-negative")
        if math.hypot(self.start[0] - self.destination[0], self.start[1] - self.destination[1]) <= self.tolerance:
            raise ValueError("start and destination must be different (farther apart than the tolerance)")
        arena = Arena(obstacles=tuple(Obstacle(*o) for o in self.obstacles), safety_margin=self.safety_margin)
        for name, p in (("start", self.start), ("destination", self.destination)):
            if not arena.inside(*p):
                limit = arena.half_size - arena.fly_radius - arena.safety_margin
                raise ValueError(f"{name} is outside the arena (limit +/-{limit:.2f})")
            if arena.hits_obstacle(*p):
                raise ValueError(f"{name} is inside an obstacle")


class FlySimulation:
    def __init__(self, config: SimConfig | None = None) -> None:
        self.status = "ready"
        self.message = "Ready"
        self.configure(config or SimConfig())

    # ---- setup -------------------------------------------------------
    def configure(self, config: SimConfig) -> None:
        config.validate()
        self.config = config
        self.arena = Arena(obstacles=tuple(Obstacle(*o) for o in config.obstacles), safety_margin=config.safety_margin)
        # Extra planning buffer absorbs the motor model's finite turning radius;
        # the movement gate itself still uses the configured arena geometry.
        self.route = plan_route(self._planning_arena(), config.start, config.destination) if config.mode == "planned" else []
        self.provenance = PROVENANCE
        if config.mode == "connectome":
            self.circuit = build_dna02_circuit(DT)
            self.provenance = DNA02_PROVENANCE
            output_nodes = self.circuit.output_nodes
        else:
            self.circuit = build_synthetic_nav_circuit(DT)
            output_nodes = OUTPUT_NODES
        self.encoder = NavigationSensoryEncoder(SENSORY_TARGETS)
        self.decoder = SteeringDecoder(output_nodes, SteeringDecoderParameters(rate_window_steps=RATE_WINDOW_STEPS))
        self.motor = MotorModel()
        self.reset()

    def reset(self) -> None:
        c = self.config
        if c.mode == "planned":
            self.route = plan_route(self._planning_arena(), c.start, c.destination)
        self.circuit.reset()
        self.decoder.reset()
        self.motor.reset(position=(c.start[0], 0.0, c.start[1]), yaw_radians=math.radians(c.yaw_deg))
        self.steps = 0
        self.sim_time = 0.0
        self.collision: str | None = None
        self.reached = False
        self.heading_error = 0.0
        self.last_cmd = {"left": 0.0, "right": 0.0, "forward": 0.0}
        self.spike_counts = {n: 0 for n in self.circuit.neurons}
        self.last_potentials = {n: 0.0 for n in self.circuit.neurons}
        self.last_spikes = {n: False for n in self.circuit.neurons}
        self.path_length = 0.0
        self.trajectory = [c.start]
        self.blocked_steps = 0
        self.route_index = 1
        self.route_replans = 0
        self._avoidance_turn = 1
        self.status, self.message = "ready", "Ready"

    def _planning_arena(self) -> Arena:
        # Keep the *legal wall boundary* identical to the movement arena. The
        # extra planning clearance should enlarge obstacle footprints only;
        # adding it to safety_margin alone also shrinks the walls and can reject
        # valid user start positions near an arena edge.
        extra_clearance = 0.35
        return Arena(half_size=self.arena.half_size + extra_clearance, fly_radius=self.arena.fly_radius,
                     safety_margin=self.arena.safety_margin + 0.35,
                     obstacles=self.arena.obstacles)

    # ---- control -----------------------------------------------------
    def start(self) -> None:
        if self.config.mode == "planned" and not self.route:
            self.status, self.message = "no_route", "No valid route"
            return
        if self.status in ("reached", "collision", "timeout"):
            self.reset()
            if self.config.mode == "planned" and not self.route:
                self.status, self.message = "no_route", "No valid route"
                return
        self.status, self.message = "running", "Running"

    def pause(self) -> None:
        if self.status == "running":
            self.status, self.message = "paused", "Paused"

    def resume(self) -> None:
        if self.status == "paused":
            self.status, self.message = "running", "Running"

    # ---- one step ----------------------------------------------------
    def step(self) -> None:
        if self.status not in ("running", "avoiding"):
            return  # paused / finished / ready: state never advances
        c = self.config
        if c.mode == "planned":
            self._step_planned()
            return
        pos = np.array(self.motor.position)
        goal = np.array([c.destination[0], 0.0, c.destination[1]])
        heading = heading_vector_from_yaw(self.motor.yaw_radians)

        if c.mode in ("neural", "connectome"):
            sensed = self.arena.sensed_points(pos[0], pos[2])
            enc = self.encoder.encode(pos, heading, goal, sensed)
            self.heading_error = enc.heading_error_radians
            res = (self.circuit.step_encoded(enc) if c.mode == "connectome"
                   else self.circuit.step(enc.currents))
            self.last_potentials, self.last_spikes = dict(res.membrane_potentials), dict(res.spikes)
            for n, s in res.spikes.items():
                self.spike_counts[n] += int(s)
            cmd = self.decoder.decode(self.circuit).as_dict()
        else:  # direct baseline: NOT neural, clearly labelled
            err = signed_heading_error(heading, goal - pos)
            self.heading_error = err
            cmd = {"left": max(0.0, -math.sin(err)), "right": max(0.0, math.sin(err)),
                   "forward": max(0.0, math.cos(err))}
        self.last_cmd = cmd

        before = np.array(self.motor.position)
        st = self.motor.step(cmd)
        hit = self.arena.segment_collision((before[0], before[2]), (st.position[0], st.position[2]))
        if hit:
            # Reject the unsafe displacement before publishing it. Keep turning so
            # the next sensing step can find a clear direction around the obstacle.
            self.motor.position = tuple(float(v) for v in before)
            self.motor.forward_velocity = 0.0
            self.blocked_steps += 1
            self.collision = None
            self.message = f"Route blocked ({hit}); turning to find a safe path"
            self.status = "avoiding"
            if self.blocked_steps % 30 == 0:
                self._avoidance_turn = -self._avoidance_turn
            self.motor.yaw_rate = self._avoidance_turn * self.motor.parameters.maximum_yaw_rate
            self.motor.yaw_radians = (self.motor.yaw_radians + self.motor.yaw_rate * DT + math.pi) % (2 * math.pi) - math.pi
            self.motor.step_index += 1
        else:
            self.blocked_steps = 0
            self.status = "running"
            self.path_length += float(np.linalg.norm(np.array(st.position) - before))
            self.trajectory.append((st.position[0], st.position[2]))
        self.steps += 1
        self.sim_time += DT

        if self.distance() <= c.tolerance:
            self.reached, self.status, self.message = True, "reached", "Destination reached"
        elif self.sim_time >= c.timeout_s:
            self.status, self.message = "timeout", "Trial failed (timeout)"

    def _step_planned(self) -> None:
        """Follow A* waypoints through the same motor and swept collision gate."""
        c = self.config
        px, _, pz = self.motor.position
        if self.distance() <= c.tolerance:
            self.reached, self.status, self.message = True, "reached", "Destination reached"
            return
        while self.route_index < len(self.route) - 1 and math.dist((px, pz), self.route[self.route_index]) <= 0.05:
            self.route_index += 1
        target = self.route[min(self.route_index, len(self.route) - 1)]
        desired_yaw = math.atan2(target[0] - px, -(target[1] - pz))
        err = (desired_yaw - self.motor.yaw_radians + math.pi) % (2 * math.pi) - math.pi
        self.heading_error = err
        turn = max(-1.0, min(1.0, err / math.pi))
        # Turn before translating. Forward velocity decays through motor dynamics,
        # so the low approach speed gives that braking drift room to settle.
        forward = 0.35 if abs(err) < 0.12 else 0.0
        cmd = {"left": max(0.0, -turn), "right": max(0.0, turn), "forward": forward}
        self.last_cmd = cmd
        before = self.motor.position
        st = self.motor.step(cmd)
        hit = self.arena.segment_collision((before[0], before[2]), (st.position[0], st.position[2]))
        if hit:
            self.motor.position = before
            self.motor.forward_velocity = 0.0
            self.blocked_steps += 1
            if self.blocked_steps >= 3:
                self.status, self.message = "route_blocked", f"Route blocked ({hit})"
            else:
                self.status, self.message = "avoiding", f"Replanning after blocked waypoint ({hit})"
                self.route = plan_route(self._planning_arena(), (before[0], before[2]), c.destination)
                self.route_index = 1
                self.route_replans += 1
                if not self.route:
                    self.status, self.message = "no_route", "No valid route"
        else:
            self.status, self.message = "running", "Following A* waypoint route"
            self.blocked_steps = 0
            self.path_length += math.dist((before[0], before[2]), (st.position[0], st.position[2]))
            self.trajectory.append((st.position[0], st.position[2]))
        self.steps += 1
        self.sim_time += DT
        if self.distance() <= c.tolerance:
            self.reached, self.status, self.message = True, "reached", "Destination reached"
        elif self.sim_time >= c.timeout_s:
            self.status, self.message = "timeout", "Trial failed (timeout)"

    def distance(self) -> float:
        p, d = self.motor.position, self.config.destination
        return math.hypot(p[0] - d[0], p[2] - d[1])

    def _neural_view(self) -> dict:
        """State shown in the UI. The connectome circuit has 107 neurons, so it is shown as
        named groups (mean potential, any-spike, total spikes); the synthetic one per neuron."""
        groups = getattr(self.circuit, "display_groups", None)
        if not groups:
            return {"provenance": self.provenance["kind"], "potentials": self.last_potentials,
                    "spikes": self.last_spikes, "spike_counts": self.spike_counts}
        pot, spk, cnt = {}, {}, {}
        for g, names in groups.items():
            pot[g] = sum(self.last_potentials[n] for n in names) / len(names)
            spk[g] = any(self.last_spikes[n] for n in names)
            cnt[g] = sum(self.spike_counts[n] for n in names)
        return {"provenance": self.provenance["kind"], "potentials": pot, "spikes": spk,
                "spike_counts": cnt, "grouped": True}

    # ---- snapshot for Godot -----------------------------------------
    def snapshot(self, seq: int) -> dict:
        s, c = self.motor.state, self.config
        return {
            "schema": SCHEMA, "seq": seq, "status": self.status, "message": self.message,
            "mode": c.mode, "controller_note": ({
                "neural": "LIF circuit (SYNTHETIC, not real connectome)",
                "connectome": "LIF circuit wired from MaleCNS v1.0 DNa02 subgraph (forward channel synthetic)",
                "direct": "direct rule baseline (no neurons)"}[c.mode] if c.mode != "planned"
                else "A* / waypoint navigation (non-neural planner)"),
            "step": self.steps, "time": round(self.sim_time, 3),
            "position": [s.position[0], s.position[1], s.position[2]],
            "yaw_rad": s.yaw_radians, "yaw_rate": s.yaw_rate, "speed": s.forward_velocity,
            "destination": [c.destination[0], 0.0, c.destination[1]],
            "start": [c.start[0], 0.0, c.start[1]],
            "distance": self.distance(), "heading_error": self.heading_error,
            "goal_tolerance": c.tolerance, "fly_radius": self.arena.fly_radius,
            "safety_margin": self.arena.safety_margin,
            "steering": self.last_cmd, "collision": self.collision, "reached": self.reached,
            "path_length": self.path_length,
            "route": [list(p) for p in self.route], "route_index": self.route_index,
            "route_replans": self.route_replans,
            "obstacles": [list(o) for o in c.obstacles],
            "neural": self._neural_view(),
        }
