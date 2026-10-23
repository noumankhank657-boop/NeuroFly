"""Local TCP server: newline-delimited JSON between Python (brain) and Godot (3D view).

Host 127.0.0.1 only. Godot -> Python commands, one JSON object per line:
  {"cmd":"configure","start":[x,z],"yaw_deg":0,"destination":[x,z],"mode":"neural|direct","obstacles":[[x,z,r],..]}
  {"cmd":"start"} {"cmd":"pause"} {"cmd":"resume"} {"cmd":"reset"} {"cmd":"ping"}
Python -> Godot: one state snapshot per line (schema nuerofly.state.v2) ~30 times/s.
Errors never crash the server: they are sent as {"schema":..., "error":"..."} and shown in the UI.
"""
from __future__ import annotations
import argparse, asyncio, json, logging, sys

from .loop import DT, SCHEMA, FlySimulation, SimConfig

log = logging.getLogger("nuerofly.server")
TICK_HZ = 30


class Server:
    def __init__(self) -> None:
        self.sim = FlySimulation()
        self.seq = 0
        self.speed = 1          # simulation-time multiplier (1..8)
        self.clients: set[asyncio.StreamWriter] = set()
        self.error: str | None = None

    def handle(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        log.info("Godot command: %s (mode=%s)", cmd, msg.get("mode", "n/a"))
        if cmd == "configure":
            d = SimConfig()
            cfg = SimConfig(
                start=tuple(msg.get("start", d.start)), yaw_deg=msg.get("yaw_deg", d.yaw_deg),
                destination=tuple(msg.get("destination", d.destination)), mode=msg.get("mode", d.mode),
                obstacles=[tuple(o) for o in msg.get("obstacles", [])],
                tolerance=msg.get("tolerance", d.tolerance), timeout_s=msg.get("timeout_s", d.timeout_s),
                safety_margin=msg.get("safety_margin", d.safety_margin))
            self.sim.configure(cfg)
            self.speed = max(1, min(8, int(msg.get("speed", self.speed))))
        elif cmd == "start": self.sim.start()
        elif cmd == "pause": self.sim.pause()
        elif cmd == "resume": self.sim.resume()
        elif cmd == "reset": self.sim.reset()
        elif cmd == "speed": self.speed = max(1, min(8, int(msg.get("value", 1))))
        elif cmd == "ping": pass
        else: raise ValueError(f"unknown command: {cmd!r}")

    async def client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.clients.add(writer)
        log.info("Godot connected")
        try:
            while line := await reader.readline():
                try:
                    self.handle(json.loads(line))
                    self.error = None
                except Exception as exc:   # report, never swallow
                    self.error = f"{type(exc).__name__}: {exc}"
                    log.warning("bad command: %s", self.error)
        except (ConnectionError, asyncio.IncompleteReadError) as exc:
            log.info("Godot connection closed: %s", exc)
        finally:
            self.clients.discard(writer)
            writer.close()
            log.info("Godot disconnected")

    async def ticker(self) -> None:
        per_tick = max(1, round(1.0 / (DT * TICK_HZ)))   # 3 steps per tick = real time
        while True:
            try:
                for _ in range(per_tick * self.speed):
                    self.sim.step()
            except Exception as exc:
                self.sim.status, self.sim.message = "error", "Simulation error"
                self.error = f"{type(exc).__name__}: {exc}"
                log.exception("simulation error")
            self.seq += 1
            snap = self.sim.snapshot(self.seq)
            snap["error"] = self.error
            data = (json.dumps(snap) + "\n").encode()
            for w in list(self.clients):
                try:
                    w.write(data)
                    await asyncio.wait_for(w.drain(), timeout=0.5)
                except (ConnectionError, OSError, asyncio.TimeoutError):
                    self.clients.discard(w)
            await asyncio.sleep(1.0 / TICK_HZ)


async def main_async(port: int) -> None:
    srv = Server()
    server = await asyncio.start_server(srv.client, "127.0.0.1", port)
    log.info("NueroFly backend listening on 127.0.0.1:%d (schema %s)", port, SCHEMA)
    async with server:
        await asyncio.gather(server.serve_forever(), srv.ticker())


def main() -> int:
    p = argparse.ArgumentParser(description="NueroFly backend server")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(main_async(a.port))
    except KeyboardInterrupt:
        log.info("stopped")
    except OSError as exc:
        print(f"Cannot start backend on port {a.port}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
