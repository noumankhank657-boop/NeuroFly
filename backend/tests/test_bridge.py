from pathlib import Path
import tempfile
import unittest
import json
from backend.bridge import write_motor_state
from backend.motor import MotorState

class BridgeTests(unittest.TestCase):
    def test_motor_state_is_published_with_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_motor_state(MotorState((1,2,3), .4, .2, .8, 9), Path(d)/"state.json")
            data = json.loads(p.read_text())
            self.assertEqual(data["schema"], "nuerofly.motor-state.v1")
            self.assertEqual(data["position"], [1,2,3])
            self.assertEqual(data["step_index"], 9)

if __name__ == "__main__": unittest.main()
