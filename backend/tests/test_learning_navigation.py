import json
import math
from pathlib import Path

from backend.navigation.learning import NavigationLearner
from backend.sim.loop import FlySimulation, SimConfig
from backend.sim.server import Server

TEST_MEMORY = Path(__file__).resolve().parents[2] / ".task_tmp" / "learning_memory_tests"


def memory_file(name):
    TEST_MEMORY.mkdir(parents=True, exist_ok=True)
    path = TEST_MEMORY / name
    path.unlink(missing_ok=True)
    return path


def test_state_is_compact_and_seed_repeats_actions():
    a = NavigationLearner(memory_file("a.json"), seed=41)
    b = NavigationLearner(memory_file("b.json"), seed=41)
    s = a.state(0.0, 0.0, 0.0, False, False)
    assert len(s.split(":")) == 4
    assert [a.choose(s) for _ in range(20)] == [b.choose(s) for _ in range(20)]
    c = NavigationLearner(memory_file("c.json"), seed=42)
    assert [a.choose(s) for _ in range(20)] != [c.choose(s) for _ in range(20)]


def test_q_update_rewards_progress_and_penalizes_failure():
    agent = NavigationLearner(memory_file("q.json"), seed=1)
    s = agent.state(0, 0, 0, False, False)
    agent.update(s, 0, 1.0, s)
    positive = agent.q[s][0]
    agent.update(s, 1, -2.0, s)
    assert positive > 0 > agent.q[s][1]


def test_memory_save_reload_and_clear():
    path = memory_file("memory.json")
    a = NavigationLearner(path, seed=3)
    s = a.state(0.2, 0.1, 0, True, False)
    a.update(s, 2, 3.0, s)
    a.episodes = 2
    a.save()
    assert json.loads(path.read_text())["version"] == 1
    b = NavigationLearner(path, seed=9)
    assert b.episodes == 2 and b.q == a.q
    b.clear()
    assert not path.exists() and b.q == {} and b.episodes == 0


def test_learning_mode_keeps_collision_gate_and_reaches_open_goal():
    sim = FlySimulation(SimConfig(start=(0, 5), destination=(0, -5), mode="learning",
                                  seed=8, timeout_s=30), memory_path=memory_file("open.json"))
    sim.learner.clear()
    sim.start()
    for _ in range(3000):
        sim.step()
        if sim.status in ("reached", "stuck", "failed"):
            break
    assert sim.status == "reached"
    assert sim.arena.collision(sim.motor.position[0], sim.motor.position[2]) is None
    assert sim.learner.episodes == 1
    assert math.isfinite(sim.distance())


def test_blocked_goal_fails_safely_and_reset_preserves_memory():
    barrier = [(float(x), 0.0, 0.8) for x in range(-10, 11, 2)]
    sim = FlySimulation(SimConfig(start=(0, 8), destination=(0, -8), mode="learning",
                                  obstacles=barrier, seed=2, timeout_s=18), memory_path=memory_file("barrier.json"))
    sim.learner.clear()
    sim.start()
    for _ in range(2000):
        sim.step()
        if sim.status in ("reached", "stuck", "failed"):
            break
    assert sim.status in ("stuck", "failed")
    assert not sim.reached and sim.collision_count > 0
    assert sim.arena.collision(sim.motor.position[0], sim.motor.position[2]) is None
    learned = {key: values[:] for key, values in sim.learner.q.items()}
    sim.reset()
    assert sim.status == "ready"
    assert sim.motor.position == (0.0, 0.0, 8.0)
    assert sim.learner.q == learned


def test_pause_and_resume_work_across_learning_statuses():
    sim = FlySimulation(SimConfig(mode="learning", seed=4), memory_path=memory_file("pause.json"))
    sim.start()
    for _ in range(30):
        sim.step()
    assert sim.status in ("exploring", "navigating", "learning", "avoiding")
    sim.pause()
    paused = (sim.steps, sim.motor.position)
    assert sim.status == "paused"
    for _ in range(30):
        sim.step()
    assert (sim.steps, sim.motor.position) == paused
    sim.resume()
    assert sim.status == "exploring"
    sim.step()
    assert sim.steps == paused[0] + 1


def test_learning_timeout_is_reported_as_failure():
    sim = FlySimulation(SimConfig(mode="learning", timeout_s=0.01),
                        memory_path=memory_file("timeout.json"))
    sim.start()
    sim.step()
    assert sim.status == "failed"
    assert not sim.reached and sim.learner.episodes == 1


def test_clear_memory_command_resets_and_deletes_persisted_table():
    server = Server()
    path = memory_file("server-clear.json")
    server.sim.memory_path = path
    server.sim.configure(SimConfig(mode="learning"))
    state = server.sim.learner.state(0, 0, 0, False, False)
    server.sim.learner.update(state, 0, 2, state)
    server.sim.learner.episodes = 1
    server.sim.learner.save()
    server.sim.start()
    server.sim.step()
    server.handle({"cmd": "clear_memory"})
    assert not path.exists()
    assert server.sim.status == "ready" and server.sim.steps == 0
    assert server.sim.motor.position == (0.0, 0.0, 7.0)
    assert server.sim.learner.q == {} and server.sim.learner.episodes == 0
