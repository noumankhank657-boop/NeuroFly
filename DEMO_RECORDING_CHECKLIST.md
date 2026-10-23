# Digital Fly recording checklist

## Before recording

- Install Python 3.10+ and Godot 4.7.
- In PowerShell at the repository root, run:

  ```powershell
  Set-ExecutionPolicy -Scope Process Bypass
  .\run_nuerofly.ps1 -GodotPath "C:\path\to\Godot_v4.7-stable_win64.exe"
  ```

- Wait until Godot shows **Backend connected**. Use the default integer seed or choose one and record it.
- In OBS or another recorder, capture the Godot window at 1280×720 or higher. Make a short test recording and verify the full UI is visible and audio is clear if narration is used.
- Avoid unrelated windows and notifications. Start the recording after the backend connection succeeds.

## Suggested 2–3 minute sequence

1. **Introduce (20–30 s):** “Digital Fly is a 3D navigation simulation. Godot visualizes a planar arena, while Python runs an experimental tabular Q-learning controller.”
2. **Run (40–60 s):** show the start point, destination, obstacles, active controller, seed and telemetry. Press Start and let the default course run. Describe the status actually shown.
3. **Controls and memory (30–40 s):** Pause and point to the unchanged step/time, Resume, then Reset. Explain that Reset keeps the learned Q-table. If useful, press **Clear learned memory + reset** and show the episode/state counters reset.
4. **Research context (30–40 s):** state that the evaluation showed reliable open-space results but mixed results with obstacles. Describe the separate DNa02 mode as experimental connectome-derived connectivity with synthetic sensor mapping and forward-speed input.
5. **Limits (10–20 s):** mention simulator-computed geometric cues, planar movement, circular obstacles and simplified motor dynamics. Do not describe the controller as a biological fly brain or flight model.

If the simulation ends in failure, show and explain the real failure. The UI reports “Destination reached” only when the distance is within the configured tolerance.

## Useful genuine screenshots

- Arena before Start, with start, destination and obstacle layout visible.
- Active simulation with the trajectory, controller label and live telemetry.
- Actual terminal result with distance and goal tolerance visible, whether reached or failed.
- Optional memory-cleared/reset screen showing zero episodes and states.

Do not manufacture or edit a success state into a screenshot. Keep the source manifest and attribution with any later public release. No webpage or video has been published by this milestone.
