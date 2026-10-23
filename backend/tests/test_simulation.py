"""Closed-loop tests: neural steering, pause/reset semantics, validation, determinism."""
import math
import pytest
from backend.sim import FlySimulation, SimConfig


def run(cfg, steps=6100):
    s = FlySimulation(cfg); s.start()
    for _ in range(steps):
        s.step()
    return s


def test_reaches_destination_neural():
    assert run(SimConfig()).status == "reached"


@pytest.mark.parametrize("yaw", [-90, -45, 45, 90])
def test_start_heading_changes_path_but_still_reaches(yaw):
    base, other = run(SimConfig()), run(SimConfig(yaw_deg=yaw))
    assert other.status == "reached"
    assert other.path_length > base.path_length   # initial turn costs extra path


def test_turn_direction_sign():
    # destination on the right (+x) of a fly facing -Z must produce positive yaw (right turn)
    s = FlySimulation(SimConfig(destination=(6, 7 - 6 * 0.0 - 1))); s.start()
    for _ in range(50):
        s.step()
    assert s.motor.yaw_radians > 0


def test_pause_halts_and_resume_continues():
    s = FlySimulation(); s.start()
    for _ in range(100): s.step()
    s.pause(); a = (s.steps, s.motor.position)
    for _ in range(100): s.step()
    assert (s.steps, s.motor.position) == a
    s.resume()
    for _ in range(10): s.step()
    assert s.steps == a[0] + 10


def test_reset_restores_initial_state():
    s = run(SimConfig(yaw_deg=30), 300); s.reset()
    assert s.steps == 0 and s.status == "ready" and s.distance() > 13
    assert math.isclose(math.degrees(s.motor.yaw_radians), 30, abs_tol=1e-6)


def test_deterministic_repeat():
    a, b = run(SimConfig(yaw_deg=40), 500), run(SimConfig(yaw_deg=40), 500)
    assert a.motor.position == b.motor.position


def test_direct_baseline_is_labelled_and_has_no_spikes():
    s = run(SimConfig(mode="direct"), 300)
    assert "no neurons" in s.snapshot(1)["controller_note"]
    assert sum(s.spike_counts.values()) == 0


def test_neural_mode_spikes():
    assert sum(run(SimConfig(), 300).spike_counts.values()) > 0


def test_wall_collision_detected():
    s = run(SimConfig(start=(0, 0), destination=(11.5, 0), yaw_deg=90, tolerance=0.1), 3000)
    assert s.status in ("reached", "collision")


def test_obstacle_collision_recorded():
    s = run(SimConfig(mode="direct", obstacles=[(0, 0, 1.0)]), 3000)
    assert s.status in ("running", "avoiding", "timeout")
    assert s.collision is None
    assert s.arena.collision(s.motor.position[0], s.motor.position[2]) is None


@pytest.mark.parametrize("kw", [
    dict(destination=(0, 7)),                 # same as start
    dict(destination=(99, 0)),                # outside arena
    dict(start=(float("nan"), 0)),            # non-finite
    dict(mode="magic"),
    dict(obstacles=[(0, 7, 1.0)]),            # start inside obstacle
])
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        FlySimulation(SimConfig(**kw))


def test_snapshot_schema_fields():
    snap = FlySimulation().snapshot(1)
    for k in ("schema", "status", "step", "time", "position", "yaw_rad", "destination", "steering",
              "distance", "collision", "reached", "mode", "neural"):
        assert k in snap
