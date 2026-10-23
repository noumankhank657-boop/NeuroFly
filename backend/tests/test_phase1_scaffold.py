import json, subprocess, sys
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[2]
class Phase1ScaffoldTests(unittest.TestCase):
    def test_default_config_is_phase_nine_motor_model(self):
        config = json.loads((ROOT / "experiments/configs/default.json").read_text())
        self.assertEqual(config["phase"], 9)
        self.assertFalse(config["controller_training"])
        self.assertEqual(config["status"], "nuerofly-motor-model")
    def test_runner_reports_verified_boundary(self):
        result = subprocess.run([sys.executable, "run_simulation.py"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NueroFly navigation math, LIF neuron, toy graph, connectome pipeline, adapter, sensory encoding, decoder, and motor verified", result.stdout)
        self.assertIn("LIF CIRCUIT", result.stdout)
