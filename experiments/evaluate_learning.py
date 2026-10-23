"""Run repeatable seeded learning-controller scenarios and print episode metrics."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from backend.sim.loop import FlySimulation, SimConfig


def run(obstacles, seed: int, memory_path: Path, timeout_s: float = 18.0,
        clear: bool = True) -> dict:
    sim = FlySimulation(SimConfig(start=(0.0, 8.0), destination=(0.0, -8.0),
                                  mode="learning", obstacles=obstacles, seed=seed,
                                  timeout_s=timeout_s), memory_path=memory_path)
    if clear:
        sim.learner.clear()
    sim.start()
    actions = []
    previous = -1
    for _ in range(round(timeout_s / 0.01) + 2):
        sim.step()
        if sim._action_age == 1 and sim.steps != previous:
            actions.append(sim.learner and sim.snapshot(0)["learning"]["action"])
            previous = sim.steps
        if sim.status in ("reached", "stuck", "failed"):
            break
    return {"status": sim.status, "steps": sim.steps, "distance": round(sim.distance(), 2),
            "blocked_moves": sim.collision_count, "path": round(sim.path_length, 2),
            "first_actions": actions[:5], "learned_states": len(sim.learner.q),
            "episodes": sim.learner.episodes}


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    return {
        "episodes": n,
        "successes": sum(item["status"] == "reached" for item in episodes),
        "failures": sum(item["status"] in ("failed", "stuck") for item in episodes),
        "collisions": sum(item["blocked_moves"] for item in episodes),
        "mean_path": round(sum(item["path"] for item in episodes) / n, 2),
        "episode_results": episodes,
    }


def main() -> None:
    layouts = {
        "open": [],
        "one_obstacle": [(0.0, 0.0, 1.5)],
        "multiple_ordinary": [(-1.2, 4.0, 0.8), (1.1, 0.0, 0.9), (-1.0, -4.0, 0.8)],
    }
    report = {}
    # Isolate each scenario's persistent Q table from the user's live demo memory.
    with tempfile.TemporaryDirectory(prefix="digital-fly-evaluation-") as temp:
        root = Path(temp)
        for name, obstacles in layouts.items():
            report[name] = {}
            for seed in (2, 11, 29):
                memory = root / name / str(seed) / "memory.json"
                runs = [run(obstacles, seed, memory, clear=(episode == 0)) for episode in range(4)]
                report[name][str(seed)] = summarize(runs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
