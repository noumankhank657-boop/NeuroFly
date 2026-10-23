"""End-to-end safety checks for deterministic A* waypoint navigation."""
from backend.sim.loop import FlySimulation, SimConfig


def run_planned(obstacles=(), **kwargs):
    sim = FlySimulation(SimConfig(mode="planned", obstacles=list(obstacles), timeout_s=90, **kwargs))
    sim.start()
    for _ in range(12000):
        if sim.status not in ("running", "avoiding"):
            break
        sim.step()
    return sim


def test_clear_route_reaches_goal_inside_tolerance():
    sim = run_planned()
    assert sim.status == "reached"
    assert sim.distance() <= sim.config.tolerance
    assert len(sim.route) >= 2


def test_single_obstacle_directly_in_route_is_circumvented_safely():
    sim = run_planned([(0, 0, 1.5)])
    assert sim.status == "reached", (sim.status, sim.distance())
    assert sim.distance() <= sim.config.tolerance
    assert all(sim.arena.collision(x, z) is None for x, z in sim.trajectory)


def test_multiple_obstacles_require_detour_and_are_reached():
    obstacles = [(0, 0, 1.5), (-1.7, -4.2, 1.0), (1.8, 3.5, 0.9)]
    sim = run_planned(obstacles)
    assert sim.status == "reached", (sim.status, sim.distance())
    assert sim.path_length > 14
    assert sim.route_replans == 0
    assert all(sim._planning_arena().segment_collision(a, b) is None
               for a, b in zip(sim.route, sim.route[1:]))


def test_narrow_open_corridor_is_supported():
    sim = run_planned([(0, 0, 2.0)], start=(3.0, 7.0), destination=(3.0, -7.0))
    assert sim.status == "reached", (sim.status, sim.distance())


def test_fully_blocked_goal_reports_no_route():
    obstacles = [(-6, 0, 5.7), (0, 0, 5.7), (6, 0, 5.7)]
    sim = FlySimulation(SimConfig(mode="planned", obstacles=obstacles))
    sim.start()
    assert sim.status == "no_route"
    assert sim.message == "No valid route"
    assert sim.reached is False


def test_following_waypoints_never_crosses_obstacle_or_arena_boundary():
    sim = run_planned([(0, 0, 1.5)])
    points = sim.trajectory
    assert points
    assert all(sim.arena.collision(x, z) is None for x, z in points)
    assert sim.status == "reached"


def test_goal_success_is_only_reported_within_tolerance():
    sim = FlySimulation(SimConfig(mode="planned", tolerance=0.25))
    sim.start()
    while sim.status in ("running", "avoiding"):
        sim.step()
    assert sim.status == "reached"
    assert sim.distance() <= 0.25


def test_reset_restores_clean_planned_run():
    sim = run_planned([(0, 0, 1.5)])
    sim.reset()
    assert sim.status == "ready" and sim.steps == 0
    assert sim.motor.position == (0.0, 0.0, 7.0)
    assert sim.distance() == 14.0
    assert sim.route_index == 1 and sim.route_replans == 0
    assert sim.route[0] == sim.config.start


def test_route_near_legal_arena_boundary_is_supported():
    sim = run_planned(start=(11.4, 7.0), destination=(9.0, 7.0), yaw_deg=-90)
    assert sim.status == "reached", (sim.status, sim.distance())
    assert all(sim.arena.inside(x, z) for x, z in sim.trajectory)
    assert all(sim.arena.segment_collision(a, b) is None
               for a, b in zip(sim.trajectory, sim.trajectory[1:]))


def test_reset_then_second_planned_run_reaches_again():
    sim = run_planned([(0, 0, 1.5)])
    first_steps = sim.steps
    sim.reset()
    sim.start()
    for _ in range(12000):
        if sim.status not in ("running", "avoiding"):
            break
        sim.step()
    assert first_steps > 0
    assert sim.status == "reached"
    assert sim.distance() <= sim.config.tolerance
