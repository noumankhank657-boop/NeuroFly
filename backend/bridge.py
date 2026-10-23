"""Write motor state in a renderer-neutral JSON format consumed by Godot.

WHAT: serialize the already-computed MotorState for the visualization layer.
WHY: Godot must display backend decisions rather than create a second controller.
OUTPUT: one atomic JSON snapshot containing position, yaw, speed, and step index.
ASSUMPTION: this Phase 8 bridge is local and single-writer; networking is deferred.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from dataclasses import asdict

from .motor import MotorState


def write_motor_state(state: MotorState, path: str | Path) -> Path:
    """Atomically publish one motor snapshot so Godot never reads a half-written file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nuerofly.motor-state.v1",
        "position": list(state.position),
        "yaw_radians": state.yaw_radians,
        "yaw_rate": state.yaw_rate,
        "forward_velocity": state.forward_velocity,
        "step_index": state.step_index,
    }
    with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as tmp:
        json.dump(payload, tmp, separators=(",", ":"))
        tmp.flush()
        os.fsync(tmp.fileno())
        temporary = Path(tmp.name)
    temporary.replace(destination)
    return destination
