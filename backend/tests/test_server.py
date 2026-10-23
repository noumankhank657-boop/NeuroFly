"""End-to-end test of the TCP protocol (the same one Godot uses)."""
import json, os, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_server_roundtrip():
    port = 8799
    p = subprocess.Popen([sys.executable, "-m", "backend.sim.server", "--port", str(port)],
                         cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                         stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                s = socket.create_connection(("127.0.0.1", port)); break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("server did not start")
        f = s.makefile("rw")
        send = lambda o: (f.write(json.dumps(o) + "\n"), f.flush())
        assert json.loads(f.readline())["schema"] == "nuerofly.state.v2"
        send({"cmd": "configure", "destination": [5, -6], "yaw_deg": 30, "speed": 8}); send({"cmd": "start"})
        st, t0 = {}, time.time()
        while time.time() - t0 < 15:
            st = json.loads(f.readline())
            if st["status"] != "running" and st["status"] != "ready":
                break
        assert st["status"] == "reached"
        send({"cmd": "bogus"})
        err = None
        for _ in range(5):
            err = json.loads(f.readline())["error"]
        assert err and "unknown command" in err      # error is reported, not swallowed
        send({"cmd": "reset"})
        for _ in range(5):
            st = json.loads(f.readline())
        assert st["status"] == "ready" and st["error"] is None   # server survived and recovered
    finally:
        p.terminate()
