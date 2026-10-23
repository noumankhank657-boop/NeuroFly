"""Small Phase 8 bridge smoke test for local Godot visualization.

WHAT: run the existing MotorModel for a short deterministic trace and publish
its latest state to godot/bridge/motor_state.json.
WHY: proves the visualization boundary consumes real motor-model output.
OUTPUT: a renderer-readable state snapshot; no navigation logic is duplicated.
ASSUMPTION: the Godot project is opened from this repository directory.
"""
from pathlib import Path
from .bridge import write_motor_state
from .motor import MotorModel

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "godot" / "bridge" / "motor_state.json"


def main() -> int:
    motor = MotorModel()
    for _ in range(60):
        state = motor.step({"left": 0.0, "right": 0.12, "forward": 0.8})
    path = write_motor_state(state, STATE_FILE)
    print(f"Published backend motor state to {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
