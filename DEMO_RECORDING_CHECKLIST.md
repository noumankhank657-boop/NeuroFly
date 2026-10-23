# Demo recording and sharing checklist

## Before recording

- Use the source tree and run `python -m pytest -q`; record the exact result and date. Do not repeat historical results as a fresh test.
- Install Python 3.10+ and Godot 4.7; launch with `run_nuerofly.ps1` and confirm **CONNECTED** before recording.
- In OBS Studio, add a **Window Capture** source for Godot, set the canvas/output to 1920×1080 and 30 fps (use 1280×720 if that matches the display), select the microphone only if the narration is clear, and record a 10-second test. Confirm the test shows the full UI and has usable audio before recording the demo.
- Close unrelated apps and notifications. Start the backend and Godot using the launcher, then begin recording only after **CONNECTED** appears.
- Keep a copy of the source data manifest with the release. Confirm the official MaleCNS citation and attribution wording before publishing the data.

## Suggested 3–4 minute recording

1. **0:00–0:20 — Introduce:** “NueroFly is a planar navigation simulation with a Godot visualizer and Python controller backend.”
2. **0:20–1:30 — A* demo:** Show the default route, press Start, point out the blue planned route, green trajectory, obstacles, and goal telemetry. Wait for success or state the observed outcome accurately.
3. **1:30–2:10 — Change the course:** Reset, move/add an obstacle, and start again. Show the actual replanning or no-route message.
4. **2:10–3:00 — Controller comparison:** Select Synthetic LIF and then connectome-wired DNa02, resetting each time. Describe these as experimental reactive modes; do not imply they use vision or are a complete biological brain.
5. **3:00–3:30 — Limits:** State that motion is planar, obstacles are circular, sensory input uses exact simulator geometry, and connectome wiring/dynamics include modeling assumptions. Mention test results only after the suite has completed successfully.

## Page and source release

1. Replace the demo panel and “add link” text in `docs/index.html` with the uploaded video and source repository URLs. Keep the limitations visible. `docs/showcase.html` is a review copy; update it too if retaining both files.
2. Publish `docs/index.html` using GitHub Pages or another static host. For GitHub Pages, create a clean GitHub repository, push the prepared project, then open **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, select `main` and `/docs`, and save. GitHub requires an entry file such as `index.html` in the publishing folder; this project includes `docs/index.html`. See [GitHub's publishing-source steps](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) and [site setup instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).
3. Publish the source repository from an isolated project Git repository; this folder currently has no `.git` directory, and Git resolves to the unrelated profile repository at `C:\Users\PMLS`. Do not run `git add`, commit, or push from that profile root. Create/clone a clean repository, copy only this project’s source and data, inspect the staged file list, and exclude `.venv`, `.pytest_cache`, `.task_tmp`, credentials, and environment files. Preserve `data/connectome/dna02/manifest.json` and required attribution.
4. If providing source as a ZIP, include the project code, Godot project, `requirements.txt`, launcher, documentation, experiments/tests, and attributed connectome data. Exclude local environments, caches, `.git`, and secrets. Do not call it a self-contained app: recipients need Python and Godot.
5. Upload the recording and source to services the instructor can access. No link or deployment is ready until its URL has been opened in a private browser window and, ideally, a second device/account.

## Link verification

- Open the showcase from a signed-out/private browser session.
- Play the video from beginning to end; check sound, captions if used, and sharing permissions.
- Open the source link and confirm repository visibility, README rendering, and the presence of the manifest and license attribution.
- Download any attached source archive, extract it to a fresh folder, and check its contents. A Windows executable is not supplied by this release.
- Send the tested links to the instructor only after these checks. Record any access restriction or untested item plainly.
