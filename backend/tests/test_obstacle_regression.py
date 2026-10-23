"""Regression tests for the NueroFly obstacle-stall bug.

Bug: saturated spiking made left == right (yaw 0) and flickering forward spikes
were rectified to speed 0 by the motor model, so the fly stopped at obstacles.
"""
from backend.connectome.synthetic_nav_circuit import build_synthetic_nav_circuit
from backend.motor import MotorModel
from backend.navigation import SteeringDecoder, SteeringDecoderParameters
from backend.connectome.synthetic_nav_circuit import OUTPUT_NODES
from backend.sim.loop import FlySimulation, SimConfig


def _rates(currents, steps=300):
    c = build_synthetic_nav_circuit(0.01)
    counts = {n: 0 for n in c.neurons}
    for _ in range(steps):
        for n, s in c.step(currents).spikes.items():
            counts[n] += int(s)
    return {n: v / steps for n, v in counts.items()}


def test_clear_left_right_difference_gives_a_clear_winner():
    # Old circuit: any input above ~0.5 saturated BOTH sides (left == right, yaw 0).
    # Now a decisive difference must survive to the output (winner-take-all).
    left = _rates({"sens_left": 0.9, "sens_right": 0.3, "sens_forward": 0.0})
    right = _rates({"sens_left": 0.3, "sens_right": 0.9, "sens_forward": 0.0})
    assert left["out_left"] > left["out_right"] + 0.5, left
    assert right["out_right"] > right["out_left"] + 0.5, right


def test_stronger_left_input_gives_stronger_left_output():
    lo = _rates({"sens_left": 0.3, "sens_right": 0.0, "sens_forward": 0.0})
    hi = _rates({"sens_left": 0.9, "sens_right": 0.0, "sens_forward": 0.0})
    assert hi["out_left"] > lo["out_left"]


def test_decoder_rate_window_removes_flicker():
    p = SteeringDecoderParameters(rate_window_steps=10)
    dec = SteeringDecoder(OUTPUT_NODES, p)
    c = build_synthetic_nav_circuit(0.01)
    fwd = []
    for _ in range(100):
        c.step({"sens_forward": 0.45})
        fwd.append(dec.decode(c).forward)
    tail = fwd[50:]
    assert max(tail) - min(tail) < 0.25, "windowed forward command should be steady"


def test_decoder_window_default_is_single_step():
    assert SteeringDecoderParameters().rate_window_steps == 1


def test_decoder_reset_clears_history():
    dec = SteeringDecoder(OUTPUT_NODES, SteeringDecoderParameters(rate_window_steps=5))
    c = build_synthetic_nav_circuit(0.01)
    c.step({"sens_left": 2.0}); dec.decode(c)
    dec.reset()
    assert len(dec._history) == 0


def test_motor_still_rectifies_flicker_by_design():
    """Documents WHY the decoder must be rate-based: raw 0.7/0 flicker gives speed 0."""
    m = MotorModel()
    for i in range(400):
        m.step({"left": 0.0, "right": 0.0, "forward": 0.7 if i % 2 == 0 else 0.0})
    assert m.forward_velocity < 0.05


def _run(obstacles, yaw=0.0, seconds=40):
    s = FlySimulation(SimConfig(obstacles=obstacles, yaw_deg=yaw, timeout_s=seconds))
    s.start()
    while s.status == "running":
        s.step()
    return s


def test_fly_gets_around_centered_obstacle():
    s = _run([(0.0, 0.0, 1.5)])
    assert s.status == "reached", (s.status, s.distance())
    assert s.collision is None


def test_fly_gets_around_left_and_right_offset_obstacles():
    for ox in (-0.8, 0.8):
        s = _run([(ox, 0.0, 1.5)])
        assert s.status == "reached", (ox, s.status, s.distance())


def test_obstacle_does_not_leave_fly_stuck():
    s = _run([(0.0, 0.0, 1.5)])
    assert s.sim_time < 30.0  # reached well before the timeout, i.e. not a stall


def test_left_right_mirror_symmetry_of_outcome():
    a = _run([(-0.8, 0.0, 1.5)]); b = _run([(0.8, 0.0, 1.5)])
    assert a.status == b.status == "reached"
    assert abs(a.path_length - b.path_length) < 1.5
