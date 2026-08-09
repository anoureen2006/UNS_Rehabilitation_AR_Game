# 5-Day Checklist v3 — Explained, Role-Organized
### Every item explains what it means, in plain language, so someone who has never seen the files can still follow along. All code for Days 1–3 is already fully built (see file-distribution-guide.md) — this checklist is about integrating, testing, and polishing it, not writing it from scratch each day.

**Roles used below:** 🎮 Unity (you) · 🖥️ Backend · 🔗 Integration · 🧠 ML · 🔄 Floater (anyone free)

---

## What the pieces are, in one paragraph each (read once, refer back as needed)

**The backend** is a small web server (built with a Python framework called FastAPI) that decides two things: *where should the next bird/star appear* and *how hard should the game be right now*. It runs on one laptop and the Unity app talks to it over Wi-Fi, the same way a phone app talks to any online service.

**AR Foundation** is Unity's toolkit for turning a phone/headset camera feed into an understanding of the real room — detecting flat surfaces (floors, tables, walls) so a virtual bird can be placed on a real one instead of floating randomly in space.

**An "anchor"** is a technique that pins a virtual object to a specific real-world point so it doesn't drift or jump around as the patient moves their head or walks — think of it like a thumbtack sticking the bird to the real room, not to the screen.

**The "difficulty controller"** is the logic (currently rule-based, not a trained AI) that watches whether the patient is hitting or missing targets and makes the game harder or easier in response — similar in spirit to how a video game raises the difficulty when you're doing well.

**"Exercise modes"** — the app has two: `bird_chase` (one moving target the patient follows) and `star_collect` (several stationary targets the patient collects, modeled on a real clinical test called the Star Cancellation Test).

---

## DAY 1 — Verify everything already built actually runs

**Goal in plain terms:** confirm the code someone (you) already wrote actually works when someone else runs it, on a real device, before building anything new on top of it.

1. 🖥️ Install the backend's dependencies and start the server. **What this means:** the backend needs certain Python libraries installed first (like installing an app before opening it), then you start it running so it's ready to answer requests. **Done when:** the terminal shows the server is running and `http://localhost:8000/docs` loads in a browser showing a list of API endpoints.
2. 🖥️ Manually test the three main backend endpoints through that `/docs` page. **What this means:** `/docs` gives you clickable buttons to test the server without writing any code — click "Try it out," fill in example values, see what comes back. **Done when:** starting a session, requesting a target, and checking metrics all return sensible JSON data, not errors.
3. 🎮 Install Unity itself and create the project, following the beginner walkthrough doc. **Done when:** Unity's Hierarchy panel (the list of objects in your scene) shows an AR Session and an XR Origin object, with no red error text in the Console panel.
4. 🎮 Import all 10 provided `.cs` script files into the project and attach them to the right objects (the walkthrough doc has exact steps). **What "attach" means:** in Unity, a script does nothing just by existing in the project — you have to drag it onto a specific object in the scene for it to actually run, similar to how a plugin does nothing until you enable it. **Done when:** every script's fields in the Inspector panel are filled in (no field says "None").
5. 🔗 Fix the `com.atteneder.gltfast` package error if it appears (edit a file called `Packages/manifest.json`, delete the line referencing it). **What this file is:** a list of external add-ons Unity downloads for your project — one of the default ones tries to fetch from the internet in a way that fails in some setups, and this project doesn't actually need it.
6. 🔗 Point the Unity app at the backend: set the `Base Url` field to `http://localhost:8000` if testing on the same computer, or the laptop's network address if testing on a phone. **Done when:** pressing Play in Unity shows "Session started" text on screen without red errors.
7. 🔗 Build the app onto a real Android phone and walk around a real room with it. **Done when:** a bird visibly appears on a real surface (table, floor) and moves.
8. 🔄 Whoever isn't blocked helps whoever is — the goal today is one working device demo, not everyone doing separate things.
9. 🔄 Send the two professor outreach emails today regardless of build status — this has the longest lead time of anything this week and doesn't depend on the code working yet.
10. 🔄 15-minute end-of-day standup: what's blocked, reassign help for tomorrow.

---

## DAY 2 — Make it robust, add ways to test without a phone, prep the expert review

11. 🎮 Add `BirdIdleAnimator.cs` to the bird prefab (a "prefab" is a reusable template object — like a stamp you can use to create many identical copies). **What this script does:** makes the bird gently bob and wiggle even when standing still, so it doesn't look like a frozen placeholder shape. **Done when:** you can visibly see the bird move slightly even before a trial starts.
12. 🎮 Confirm the anchor-health handling in `SessionManager.cs` works. **What this means:** if the phone's tracking gets confused (e.g. you cover the camera, or walk somewhere with blank white walls it can't map), the bird should quietly reposition instead of floating in the wrong spot or crashing the app — and this should NOT count as a "miss" for the patient. **Done when:** you deliberately cover the camera mid-trial and the game recovers gracefully.
13. 🖥️ Run the automated tests for the difficulty logic: `pytest test_difficulty_controller.py -v`. **What a "test" is here:** a small script that checks the difficulty logic behaves correctly (gets harder after hits, easier after misses, never goes below/above sane limits) without needing a person to manually check by hand every time. **Done when:** all 10 tests show "PASSED."
14. 🖥️ Run `simulate_client.py` to generate a fake full session. **What this does:** pretends to be the Unity app, sending realistic requests to the backend — so you can test and demo the difficulty adaptation logic without needing a phone in the room at all. **Done when:** it prints ~30 lines of simulated trials and a final metrics summary.
15. 🖥️ Open the CSV export link the simulator prints at the end in a browser. **What CSV means:** a spreadsheet-style text file (opens in Excel/Google Sheets) listing every single trial's data — hit/miss, reaction time, which side of the room, difficulty at that moment. **Done when:** the file opens and the numbers look sensible (hit trials have reasonable reaction times, difficulty values increase/decrease over time).
16. 🔗 Time how long a real request takes on the actual phone over Wi-Fi (watch the delay between turning to look at a bird and the next one appearing). **Why this matters:** it justifies to judges why a simple request-response system (not a constantly-open connection) was the right technical choice — write the number down.
17. 🔗 Deliberately stop the backend server while the phone app is running, see what happens, then confirm restarting the server lets a fresh session start cleanly.
18. 🧠 Read through `difficulty_controller.py` with the goal of understanding, not rewriting yet — it's the exact place a smarter algorithm would eventually go. See `file-distribution-guide.md` for a clear-eyed read on what's realistic to build this week versus what to describe as future work.
19. 🔄 Copy the content from `expert-review-instrument.md` into a real Google Form, and have one teammate fill it out as a practice run, timing how long it takes (should be under 20 minutes total).
20. 🔄 Confirm the expert review session time (professor visit) is locked in for Day 3 or 4 — chase this today if it isn't confirmed yet, it's the most schedule-sensitive item on the whole plan.

---

## DAY 3 — Turn on the second exercise mode (Star Cancellation) + run the expert session

21. 🎮 Build a simple star-shaped (or reuse the sphere) prefab, attach `StarTarget.cs` to it. **What this adds:** the ability for the game to spawn several targets at once that the patient collects in any order, instead of one moving bird at a time — this mirrors a real, established clinical neglect assessment.
22. 🎮 Create a `GameManager`-level object (or reuse the existing one) with `StarCollectRoundController.cs` attached, and wire its fields (spawn finder, star prefab, HUD) in the Inspector, same pattern as the bird setup.
23. 🎮 In `SessionManager.cs`'s Inspector fields, set `Exercise Mode` to `star_collect` on a **separate test build** (keep a `bird_chase` build too — you'll likely want both available for the demo) to confirm the mode switch actually spawns multiple stars instead of one bird.
24. 🖥️ No new backend work needed today — the `/session/{id}/next-round` endpoint that powers star mode already exists and was tested. If Unity's star round behaves unexpectedly, check with the backend person whether the JSON being sent matches `NextRoundRequest` in `schemas.py` before assuming it's a Unity bug.
25. 🔗 Test a full star_collect round on a real device: confirm several targets appear at once, each can be individually collected by looking at it, and the HUD's "Stars: X / Y" counter updates correctly.
26. 🔗 Confirm a round ending (either all collected, or time runs out) correctly triggers the next round to spawn, without needing to restart the app.
27. 🔄 Prepare the physical device(s), backup device, and stable Wi-Fi/hotspot for the expert reviewer's visit today.
28. 🔄 Run the expert review session: demo both exercise modes if time allows, walk through the questionnaire, ask the open-ended questions.
29. 🔄 Get the reviewer's quotable statement in writing (follow up by email same day if it wasn't captured live).
30. 🔄 Write up the results immediately while fresh: SUS score, rubric averages, verbatim quotes, and — critically — sort their feedback into "fix before Friday" versus "future work," and share that list with the whole team same evening.

---

## DAY 4 — Incorporate feedback, polish, prep pitch materials
*(Deliberately not pre-specified in file terms, since it depends entirely on what Day 3's expert session surfaces — but the categories to plan around are: implement the highest-priority fix from feedback, polish visuals/HUD legibility, finalize the metrics chart(s) for the pitch deck, and freeze all further code changes by end of day.)*

## DAY 5 — Pitch day
*(Logistics and rehearsal only — device checks, timed run-throughs, honest framing language for what's been validated versus what hasn't, and submission deadlines. No new code.)*
