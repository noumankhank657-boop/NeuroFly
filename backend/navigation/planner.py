"""Deterministic 8-connected A* route planner for the arena's X-Z plane."""
from __future__ import annotations

import heapq
import math

from backend.sim.environment import Arena


def plan_route(arena: Arena, start: tuple[float, float], goal: tuple[float, float],
               cell_size: float = 0.3) -> list[tuple[float, float]]:
    """Return collision-safe waypoints including the exact start and goal.

    Grid edges and final shortcuts are checked with Arena.segment_collision,
    sharing its fly radius, wall limits, obstacle footprints, and clearance.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if not arena.inside(*start) or not arena.inside(*goal):
        return []
    if arena.segment_collision(start, start) or arena.segment_collision(goal, goal):
        return []

    limit = arena.half_size - arena.fly_radius - arena.safety_margin
    lo = math.ceil(-limit / cell_size)
    hi = math.floor(limit / cell_size)
    source = (round(start[0] / cell_size), round(start[1] / cell_size))
    target = (round(goal[0] / cell_size), round(goal[1] / cell_size))

    def point(cell: tuple[int, int]) -> tuple[float, float]:
        return cell[0] * cell_size, cell[1] * cell_size

    def clear(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return arena.segment_collision(a, b) is None

    def heuristic(c: tuple[int, int]) -> float:
        return math.hypot(target[0] - c[0], target[1] - c[1])

    queue: list[tuple[float, float, tuple[int, int]]] = [(heuristic(source), 0.0, source)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {source: None}
    costs = {source: 0.0}
    directions = [(dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dz]
    found = None
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        if current == target:
            found = current
            break
        for dx, dz in directions:
            nxt = current[0] + dx, current[1] + dz
            if not (lo <= nxt[0] <= hi and lo <= nxt[1] <= hi):
                continue
            if not clear(point(current), point(nxt)):
                continue
            new_cost = cost + cell_size * (math.sqrt(2) if dx and dz else 1.0)
            if new_cost < costs.get(nxt, math.inf):
                costs[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(queue, (new_cost + heuristic(nxt), new_cost, nxt))
    if found is None:
        return []

    cells = []
    cursor = found
    while cursor is not None:
        cells.append(point(cursor))
        cursor = came_from[cursor]
    cells.reverse()
    raw = [start, *cells, goal]
    # Greedily remove grid turns where the direct swept path remains safe.
    route = [raw[0]]
    anchor = 0
    while anchor < len(raw) - 1:
        farthest = anchor + 1
        for candidate in range(anchor + 2, len(raw)):
            if clear(raw[anchor], raw[candidate]):
                farthest = candidate
        route.append(raw[farthest])
        anchor = farthest
    return route
