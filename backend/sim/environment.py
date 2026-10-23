"""Arena geometry and collision checks (X-Z plane; Y is up).

Coordinates: right-handed, Y up, zero yaw faces -Z, positive yaw turns RIGHT.
Units: arbitrary 'metres'. Godot uses the SAME axes, so positions copy 1:1;
only the rotation sign differs (Godot rotation.y = -yaw).
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Obstacle:
    x: float
    z: float
    radius: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.z, self.radius)) or self.radius <= 0:
            raise ValueError("obstacle needs finite x, z and radius > 0")


@dataclass
class Arena:
    half_size: float = 12.0          # walls at +/- half_size on X and Z
    fly_radius: float = 0.3          # collision radius of the fly body
    safety_margin: float = 0.15      # extra clearance for discrete movement steps
    obstacles: tuple[Obstacle, ...] = ()

    def inside(self, x: float, z: float) -> bool:
        m = self.half_size - self.fly_radius - self.safety_margin
        return abs(x) <= m and abs(z) <= m

    def hits_obstacle(self, x: float, z: float) -> bool:
        return any(math.hypot(x - o.x, z - o.z) <= o.radius + self.fly_radius for o in self.obstacles)

    def collision(self, x: float, z: float) -> str | None:
        if not self.inside(x, z):
            return "wall"
        if self.hits_obstacle(x, z):
            return "obstacle"
        return None

    def segment_collision(self, start: tuple[float, float], end: tuple[float, float]) -> str | None:
        """Return a collision along a swept circular fly path, including arena edges."""
        dx, dz = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dz)
        # Sampling spacing is at most half the configured safety margin.
        count = max(1, math.ceil(length / max(self.safety_margin * 0.5, 0.01)))
        for i in range(1, count + 1):
            t = i / count
            hit = self.collision(start[0] + dx * t, start[1] + dz * t)
            if hit:
                return hit
        # Expand obstacles by the margin for a conservative swept check.
        for o in self.obstacles:
            vx, vz = dx, dz
            denom = vx * vx + vz * vz
            t = 0.0 if denom == 0 else max(0.0, min(1.0, ((o.x-start[0])*vx + (o.z-start[1])*vz)/denom))
            if math.hypot(start[0] + t*vx - o.x, start[1] + t*vz - o.z) <= o.radius + self.fly_radius + self.safety_margin:
                return "obstacle"
        return None

    def sensed_points(self, x: float, z: float) -> list[tuple[float, float, float]]:
        """Nearest SURFACE point of every obstacle (what the fly 'sees').

        Using the surface (not the centre) means a big obstacle is sensed as
        close when its edge is close. Walls are NOT sensed (documented limit).
        """
        pts = []
        for o in self.obstacles:
            d = math.hypot(o.x - x, o.z - z)
            if d < 1e-9:
                pts.append((x, 0.0, z)); continue
            k = max(d - o.radius, 0.0) / d
            pts.append((x + (o.x - x) * k, 0.0, z + (o.z - z) * k))
        return pts
