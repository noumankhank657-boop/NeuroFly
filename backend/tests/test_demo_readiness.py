"""Deterministic outcome and collision-safety checks used by the final demo."""
from backend.sim.environment import Arena, Obstacle
from backend.sim.loop import FlySimulation, SimConfig


def finish(config, max_steps=10000):
    sim = FlySimulation(config)
    sim.start()
    for _ in range(max_steps):
        if sim.status not in ("running", "avoiding"):
            break
        sim.step()
    return sim


def test_clear_destination_reached_with_configured_tolerance():
    sim = finish(SimConfig())
    assert sim.status == "reached"
    assert sim.distance() <= sim.config.tolerance


def test_swept_path_rejects_collision_before_entering_obstacle():
    arena = Arena(obstacles=(Obstacle(0, 0, 1),))
    assert arena.segment_collision((0, 2), (0, 1.4)) == "obstacle"
    assert arena.collision(0, 1.4) is None


def test_neural_controller_clears_obstacle_and_multiple_obstacles():
    for obstacles in ([(0, 0, 1.5)], [(-1.1, 2.0, 1.1), (1.1, -1.5, 1.1)]):
        sim = finish(SimConfig(obstacles=obstacles, timeout_s=60))
        assert sim.status == "reached", (sim.status, sim.distance(), obstacles)
        assert sim.collision is None


def test_default_connectome_demo_tree_route_reaches_destination():
    sim = finish(SimConfig(mode="connectome", obstacles=[(0, 0, 1.5)]))
    assert sim.status == "reached"
    assert sim.distance() <= sim.config.tolerance


def test_near_boundary_start_stays_inside_arena():
    sim = finish(SimConfig(mode="neural", start=(11.4, 7), destination=(9, 7), yaw_deg=-90))
    assert sim.status in ("reached", "timeout")
    assert sim.arena.inside(sim.motor.position[0], sim.motor.position[2])


def test_unreachable_goal_does_not_report_success():
    sim = finish(SimConfig(mode="neural", start=(0, 7), destination=(0, -7),
                           obstacles=[(-6, 0, 5.7), (0, 0, 5.7), (6, 0, 5.7)], timeout_s=1))
    assert sim.status == "timeout"
    assert sim.reached is False
    assert sim.distance() > sim.config.tolerance


def test_reset_clears_avoidance_collision_path_and_controller_state():
    sim = FlySimulation(SimConfig(mode="neural", obstacles=[(0, 0, 1)]))
    sim.start()
    for _ in range(300):
        if sim.status not in ("running", "avoiding"):
            break
        sim.step()
    sim.reset()
    assert sim.status == "ready" and sim.steps == 0 and sim.sim_time == 0
    assert sim.collision is None and sim.reached is False and sim.path_length == 0
    assert sim.blocked_steps == 0
    assert sim.motor.position == (0.0, 0.0, 7.0)
