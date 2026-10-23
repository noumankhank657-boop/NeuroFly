"""Small tabular Q learner for local, sensory-driven navigation.

State = goal bearing (8 sectors), obstacle cue (clear/left/right/both),
near-obstacle (yes/no), and stuck (yes/no). Actions use the existing motor:
forward, forward-left, forward-right, pivot-left, pivot-right. The table learns
which action tends to reduce goal distance without collisions. Q-learning uses
Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a)).
This is an adaptive navigation controller, not a biological brain or neural net.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

MEMORY_VERSION = 1
ACTIONS = ("forward", "forward_left", "forward_right", "pivot_left", "pivot_right")
COMMANDS = (
    {"left": 0.0, "right": 0.0, "forward": 1.0},
    {"left": 0.65, "right": 0.0, "forward": 0.7},
    {"left": 0.0, "right": 0.65, "forward": 0.7},
    {"left": 0.8, "right": 0.0, "forward": 0.0},
    {"left": 0.0, "right": 0.8, "forward": 0.0},
)


class NavigationLearner:
    def __init__(self, path: str | Path, seed: int = 1, alpha: float = 0.25,
                 gamma: float = 0.9, epsilon: float = 0.12) -> None:
        self.path = Path(path)
        self.rng = random.Random(seed)
        self.seed = seed
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q: dict[str, list[float]] = {}
        self.episodes = 0
        self._load()

    @staticmethod
    def state(goal_error: float, left_cue: float, right_cue: float,
              near: bool, stuck: bool) -> str:
        sector = int(round(goal_error / (math.pi / 4))) % 8
        cue = (1 if left_cue > 0.08 else 0) | (2 if right_cue > 0.08 else 0)
        return f"{sector}:{cue}:{int(near)}:{int(stuck)}"

    def choose(self, state: str, explore: bool = True) -> int:
        values = self.q.setdefault(state, self._sensory_prior(state))
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTIONS))
        best = max(values)
        ties = [i for i, value in enumerate(values) if value == best]
        return self.rng.choice(ties)

    def update(self, state: str, action: int, reward: float, next_state: str,
               terminal: bool = False) -> None:
        old = self.q.setdefault(state, self._sensory_prior(state))[action]
        future = 0.0 if terminal else max(self.q.setdefault(next_state, self._sensory_prior(next_state)))
        self.q[state][action] = old + self.alpha * (reward + self.gamma * future - old)

    @staticmethod
    def _sensory_prior(state: str) -> list[float]:
        """Give unseen states a mild steering preference from current sensory cues."""
        try:
            goal, cue, near, _stuck = (int(v) for v in state.split(":"))
        except (TypeError, ValueError):
            return [0.0] * len(ACTIONS)
        # Sector 0 is aligned; sectors 1-3 indicate right and 5-7 left.
        turn_right = goal in (1, 2, 3)
        turn_left = goal in (5, 6, 7)
        values = [0.25, 0.0, 0.0, -0.1, -0.1]
        if turn_left:
            values[1], values[3] = 0.8, 0.45
        elif turn_right:
            values[2], values[4] = 0.8, 0.45
        # Encoder cue bit 1 means an obstacle left, bit 2 an obstacle right.
        if cue & 1:
            values[2] += 1.0
            values[4] += 0.55
        if cue & 2:
            values[1] += 1.0
            values[3] += 0.55
        if near:
            values[0] -= 0.7
            values[3] += 0.25
            values[4] += 0.25
        return values

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps({"version": MEMORY_VERSION, "episodes": self.episodes,
                                    "q": self.q}, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def clear(self) -> None:
        self.q.clear()
        self.episodes = 0
        self.path.unlink(missing_ok=True)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != MEMORY_VERSION:
                return
            table = data.get("q", {})
            if isinstance(table, dict) and all(isinstance(k, str) and isinstance(v, list)
                    and len(v) == len(ACTIONS) and all(math.isfinite(float(x)) for x in v)
                    for k, v in table.items()):
                self.q = {k: [float(x) for x in v] for k, v in table.items()}
                self.episodes = max(0, int(data.get("episodes", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.q, self.episodes = {}, 0
