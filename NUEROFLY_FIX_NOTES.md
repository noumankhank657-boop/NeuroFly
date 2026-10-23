# NueroFly obstacle-stall fix (synthetic baseline circuit)

NOTE: this fixes the SYNTHETIC baseline only. It is not connectome data.

## Cause (measured, Python backend only; live Godot not tested)
1. Saturation: with tau=0.02 s, R=4 any input above ~0.5 made a neuron fire every step.
   Left 0.728 vs right 0.677 both gave rate ~1.0 -> decoded 0.70 / 0.70 -> yaw = 0.
2. Flicker: a forward neuron firing every 2nd step gave commands 0.7, 0, 0.7, 0 ...
   MotorModel decelerates faster (6.0) than it accelerates (4.0), so speed pinned at 0
   (same average steady 0.35 command gives speed 0.70).

## Changes
- backend/navigation/decoder.py: optional `rate_window_steps` (default 1 = old behaviour)
  + `reset()`. Averages the last N steps into a firing-rate estimate.
- backend/sim/loop.py: uses `RATE_WINDOW_STEPS = 10` (0.1 s); resets decoder in reset().
- backend/connectome/synthetic_nav_circuit.py: tau 0.02 -> 0.05, R 4 -> 8 (weights unchanged).
- backend/tests/test_obstacle_regression.py: 10 new tests (7 fail on the old code).
- experiments/trace_obstacle.py: prints the sensory -> command -> motion trace.

## Results (Python backend, 30 random obstacle cases, seed 12345, 40 s timeout)
| setup | reached | collision | timeout |
|---|---|---|---|
| original | 9 | 0 | 21 |
| decoder window only | 22 | 1 | 7 |
| window + retuned circuit | 27 | 1 | 2 |
| direct rule baseline (cannot sense obstacles) | 9 | 21 | 0 |
Full test suite: 95 passed (85 old + 10 new).

## Known limits
- Rates are still stepwise (0, .2, .33, .5, 1.0). Near-equal left/right inputs
  (e.g. 0.73 vs 0.68) still give equal outputs -> yaw 0 for that moment.
- The retuned constants were picked by a sweep over 6 scenarios, then checked on the
  30 held-out cases above. They are hand-tuned, not biological.
- 1 collision and 2 timeouts remain unexplained in the 30-case run.
