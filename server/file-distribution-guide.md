# Who Gets What — File Distribution Guide

## Your actual team shape
- **You:** currently on Unity, will also help with ML/RL
- **1 dedicated Backend person**
- **1 dedicated Integration person**
- **1 dedicated ML person**
- **Floating members:** circulate across fields as needed

## The rule that fixes "only I have all the files"
Set up **one shared location today** — either a Git repo (recommended) or a shared cloud folder (Google Drive/Dropbox) — with this exact top-level structure, and everyone pulls from it instead of you individually sending files:

```
usn-ar-project/
├── backend/              <- Backend person owns this folder
├── unity-scripts/        <- goes INSIDE the Unity project's Assets/Scripts folder
├── docs/                 <- checklists, walkthroughs, this file
```

If you're not using Git yet: **do it today, not later** — a shared Drive folder works for the `backend/` Python files fine, but the Unity project itself (once it has a scene, prefabs, and imported packages) gets large and binary-heavy fast; Drive sync on binary files causes silent corruption/conflicts. Git with Git LFS (or just accepting one person "owns" the Unity project file and others get script-only diffs from them) avoids this. Floating members should default to whichever system the Unity dev (you) sets up first.

---

## Backend person — gets the `backend/` folder, nothing else required
Files: `main.py`, `schemas.py`, `session_store.py`, `difficulty_controller.py`, `spawn_filter.py`, `test_difficulty_controller.py`, `simulate_client.py`, `requirements.txt`

Give them: `docs/backend-walkthrough.md` (setup from zero, no Unity/ML knowledge assumed).

They do not need any `.cs` files at all — the backend has zero dependency on Unity to run or test (that's what `simulate_client.py` proves).

---

## You (Unity) — gets `unity-scripts/`
Files: `ARSessionManager.cs`, `SafeSpawnPointFinder.cs`, `BirdController.cs`, `BirdIdleAnimator.cs`, `GazeHeadTracker.cs`, `NetworkClient.cs`, `SessionManager.cs`, `HUDController.cs`, `StarTarget.cs`, `StarCollectRoundController.cs`

Give yourself: `docs/unity-beginner-walkthrough.md` for setup, this conversation for the "why" behind each script.

You need the backend running (even just on your own laptop via `simulate_client.py` or a real `uvicorn` instance) to test the networked parts, but you don't need to understand `difficulty_controller.py`'s internals to wire up the Unity side — you only need to know the request/response shape, which `schemas.py` defines and `NetworkClient.cs`'s DTOs already mirror exactly.

---

## Integration person — needs to see BOTH folders, but isn't editing the core logic in either
Their job is the seam between them: confirming the JSON contract actually matches (it does, as shipped — see the "Contract" note below), testing round-trip latency, handling connection failures gracefully, and being the person who can debug "is this a Unity bug or a backend bug" fastest since they understand both sides' request/response shapes.

Give them: both `backend/` and `unity-scripts/` read access, plus this note — **the contract is defined once, in `backend/schemas.py`, and mirrored in `unity-scripts/NetworkClient.cs`'s DTO classes.** If a field ever needs to change, it must change in both places in the same commit, or everything breaks silently (Unity won't error, it'll just send/receive blank/zero values for the mismatched field). This is the single most important thing for them to police all week.

---

## ML person — this is the part that needs an honest conversation, not just a file handoff
As built, there is no separate "ML file" to hand off — `difficulty_controller.py` is a rule-based staircase controller, not a trained model, and that was a deliberate Day 1 scope decision (see the original architecture doc: real RL was explicitly deferred). Here's what to actually give your ML teammate, in order of what's realistic this week:

1. **`backend/difficulty_controller.py`** — show them this first. It exposes exactly the function signature a real policy would need: `next_difficulty(current_state) -> new_difficulty_params`. Their job, if there's time, is to build an alternative to this function, not to bolt something separate on.
2. **`backend/session_store.py`** and the JSON files it produces in `backend/data/`** — this is the training/evaluation data source if they want to do anything model-based (e.g. a contextual bandit over speed/eccentricity/distance given hit-rate history). Point them at `simulate_client.py` to generate synthetic session data fast, since real device data will be scarce this week.
3. **Realistic scope for an ML person in the remaining days:** a genuinely trained RL policy needs an environment, a reward signal, and training time you likely don't have this week. The higher-value use of an ML person's time is probably: (a) formalizing what "reward" should mean for this task (e.g. balance of hit-rate ~70-80%, not too easy/hard, weighted toward neglected-side engagement) as a written spec, even if not implemented as trained RL by Friday, and (b) analyzing the simulated/real session CSVs for patterns that could inform a smarter rule-based heuristic as a middle ground between what exists now and full RL. Both of these are legitimate, honest contributions that don't require overclaiming "RL" in the pitch when what's shipped is a rule-based controller.

Since you said you'll also help with ML — this is genuinely the highest-uncertainty part of the project scope-wise, worth a 15-minute conversation with your PI or the ML teammate today to explicitly agree on what's being claimed in the pitch (rule-based adaptive difficulty, RL as documented future work) versus what would need to be true to claim "RL-based" honestly.

---

## Floating members
Give them read access to everything, but assign them per-day to whichever track's checklist (see `5-day-checklist-v2.md`) has the most unchecked items at standup — that's a better allocation signal than a fixed assignment, since it's genuinely unpredictable in advance which track will be blocked on a given day.
