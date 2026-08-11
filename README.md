# USN AR Rehab Game

An AR mobile app (Unity + AR Foundation) for patients with **Unilateral Spatial Neglect (USN)**, a post-stroke condition where patients lose awareness of one side of their surroundings. The patient turns their head to find and follow an AR bird (or collect a batch of AR stars) spawned in their real room, forcing scanning toward their neglected visual side. Difficulty adapts automatically based on hit/miss performance. Built as a 5-day hackathon project, grounded in the clinical **Visual Scanning Training (VST)** literature.

---

## How it works

Unity handles AR (plane detection, raycasting, rendering, gaze/hit detection) entirely on-device. It sends only small JSON messages — candidate spawn points and trial results — to a Python backend, which decides where the next target should appear and how hard the game should be, then sends that back. No video is streamed between them.

```
Unity (AR Foundation)  <-- REST (JSON) -->  Python backend (FastAPI)  -->  difficulty controller (rule-based, RL in progress)
```

Two exercise modes:
- **`bird_chase`** — one moving target the patient follows
- **`star_collect`** — several static targets collected in any round, modeled on the clinical Star Cancellation Test

---

## Repository layout

```
backend/            <- Python/FastAPI server, difficulty logic, tests, simulator
backend/checkpoints/ <- RL model checkpoints (in progress, see RL status below)
unity-project/       <- Unity project (Assets/script has all game logic)
docs/                <- setup walkthroughs, checklists, validation plan, evaluation plan
```

See `docs/file-distribution-guide.md` for which files matter to which role.

---

## Running the backend

```
cd backend
pip install -r requirements.txt --break-system-packages
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Confirm it's running at `http://localhost:8000/docs`. Full zero-context walkthrough: `docs/backend-onboarding-zero-context.md`.

Test without a phone:
```
python3 simulate_client.py --trials 30 --neglect-side left --hit-rate 0.7
```

Run the test suite:
```
pytest test_difficulty_controller.py -v
```

---

## Running the Unity app

Open `unity-project/` in Unity (2022 LTS or newer recommended). Full step-by-step setup, including first-time-ever-using-Unity instructions, is in:
- `docs/unity-beginner-walkthrough.md` (initial setup, AR Foundation, first working loop)
- `docs/unity-walkthrough-day2-to-day5.md` (idle animation, star_collect mode, polish)

Before building to a device, set `NetworkClient.baseUrl` (on the `GameManager` object, `Network Client` component) to your machine's current LAN IP, e.g. `http://192.168.x.x:8000` — not `localhost`.

**Android:** requires `android:usesCleartextTraffic="true"` in a custom `AndroidManifest.xml` (Player Settings → Publishing Settings), since Android blocks plain HTTP by default.
**Editor testing:** requires Player Settings → Other Settings → "Allow downloads over HTTP*" set to "Always allowed" for every relevant platform tab, and a full Editor restart after changing it.

---

## Difficulty logic / RL status

- `backend/difficulty_controller.py` — the current, working, rule-based adaptive controller (staircase logic: streaks of hits/misses adjust speed, eccentricity, distance, target count within safe bounds). This is what the app runs today.
- An RL-based controller (Stable-Baselines3 PPO) is in progress, intended as a drop-in replacement matching the same function signatures (`next_difficulty`, `next_difficulty_for_round`) so no other part of the system needs to change. See `docs/rl-module-spec.md` for the exact interface contract. **Not yet integrated** — currently missing supporting files (`unity_env.py`, `unity_dataset.py`, `unity_schema.py`) needed to load and run the trained model.

---

## Known limitations (intentional scope decisions, not bugs)

- No live camera/video streaming to the backend — Unity does all AR processing locally.
- `ARAnchor`-based world-space pinning was removed after hitting an AR Foundation compatibility issue; targets currently spawn in raw world space, sufficient for the short duration of a single trial.
- Difficulty adaptation is rule-based by default; RL integration is in progress (see above).
- No clinical efficacy claims — expert content validation (SUS + clinical-plausibility rubric) has been sought from reviewing clinicians, but this is formative feedback, not a clinical trial. See `docs/expert-validation-plan.md`.

---

