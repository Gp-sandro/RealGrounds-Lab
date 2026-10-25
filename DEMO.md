# Demo script — RealGrounds Night Shift (Track 3: Unsupervised)

**One line:** an AI agent that works unattended for hours, turning ordinary photos of real rooms into
physics-checked simulation rooms with tasks proven solvable or impossible, and that knows when it is stuck,
repeating itself, out of budget, or needs a person.

**The easiest way: the app.** Double-click `RealGrounds Lab.command` (or run `python app.py`). It opens in the
browser with four tabs:

1. **Try it:** pick a room or upload a photo, see what it measured, give the agent a task, watch it work.
2. **Stop rules:** light / heavy / wall, one click each.
3. **Unattended runs:** the long run and its re-check.
4. **Tests & results:** run the 44 tests live and show every measured number.

Use `python app.py --share` to let people on the same Wi-Fi use it from their phones and upload photos straight
from the camera. macOS may ask to allow incoming connections for Python; allow it. Anyone on the network can then
use it, and the app's own money cap (`--cap`, default $1) limits what they can spend. The written results are in
[RESULTS.md](RESULTS.md).

All commands run from `realgrounds-voice/` with the venv active:
```
cd ~/Desktop/ML-WorldModel/realgrounds-voice && source ../.venv/bin/activate
```

---

## Before you go on stage

1. **Mac plugged in.** Long runs and the live build use the GPU.
2. `python demo.py check`. Every line must be ✓ and it must end with **READY**.
3. The long run has finished (or is still running: the page refreshes itself):
   `python demo.py dashboard long1`.
4. Open these in tabs as fallbacks:
   - the long run's dashboard;
   - `outputs/night/live-rehearsal/dashboard.html`, a live run already done on our own dorm photo;
   - one replay video, for example `outputs/night/test10b/rooms/nyu_0588/tasks/2_replay.mp4`.
5. Phone: AirDrop works to this Mac, and photos land in `~/Downloads`.
6. **No long run in the background during the live upload.** An unattended run shares the GPU: a new photo
   that takes about 1 minute alone took 5 minutes next to the 240-room run. Let it finish, or stop it (Ctrl+C,
   or `kill` its process) and resume it later with the same command. The app warns you when one is running.

---

## The script (about 4 minutes)

### 0:00 — The problem (30 s)
> "Agents already run unsupervised on code, because a code mistake can be undone. In the physical world a
> mistake can't be undone, so today a human has to watch. And anyone building AI for physical spaces needs
> thousands of realistic rooms to test in. Checking each one by hand doesn't scale."

### 0:30 — The night (60 s) → the long run's dashboard
- **Top tiles:** how many real rooms, how many hours unattended, how much it cost.
- **"Working on its own":** rooms finished over time, money over time. No human input during the run.
- **"It knows when to stop":** the counts of stuck, out of budget, impossible, repeats refused, and bad proposals
  dropped. Click **one transcript** and read its last line aloud: the agent says what it tried and why it stopped.

> "It never decides that it succeeded. Physics and geometry decide. Claude only proposes. Every action is
> tried on a snapshot of the room and undone, checked bit for bit, and only an action that measurably helps
> is kept."

### 1:30 — A solution you can re-check (45 s)
```
python demo.py replay long1
```
This takes a task solved during the night and replays only the kept actions on a fresh copy of the room, with
video. The goal holds again.
> "The simulation is deterministic, so every solution in the library is a proof anyone can re-run."

### 2:15 — Live, in this room (75 s)
1. Take one photo of the room you're standing in and AirDrop it.
2. Run:
   ```
   python demo.py live ~/Downloads/IMG_XXXX.HEIC
   ```
   It takes about 1–2 minutes (the room build is about 1 minute). While it runs, say:
   > "From one ordinary photo it measures the room: depth, the lens, every object's size, and what rests on
   > what. It loads that into a physics engine. Then Claude writes tasks a person might ask a robot here, and
   > the agent tries to prove each one."
3. The dashboard opens. Show:
   - the photo next to the objects found;
   - the tasks Claude wrote;
   - which were solved, with a **video** link;
   - which were stopped, and the reason it gave.

> "We don't know what it will find in this room. That's the point: it tells us honestly."

### 3:30 — Close (30 s)
> "Two supervisors: one per task, one for the whole night. They watch money, time, repeats and failure
> streaks. Next: the same tools as an MCP server, so any AI agent can use real rooms safely, with the
> supervisor built into the server."

---

## If something goes wrong

| Problem | Do this |
|---|---|
| The live build fails or is slow | Open `outputs/night/live-rehearsal/dashboard.html`: the same flow on our dorm photo |
| No internet / API down | `python run_task.py --scene heavy --video` uses fixed rules, no API: it is stuck, says why, and asks for help. Then show the long run's dashboard (already on disk). |
| The photo is HEIC and won't open | `python demo.py live` converts HEIC with macOS `sips`; or export the photo as JPEG on the phone |
| The long run isn't finished | Its dashboard refreshes every minute; show it running ("it is working right now, unattended") |

---

## Track 3 requirements → where they show

| Track 3 asks | Where in the demo |
|---|---|
| Operates for hours without supervision | The long run's tiles and the "rooms finished" chart; `timeline.jsonl` has a timestamp on every event |
| Recognises when it is stuck | "stuck" count + transcripts ("Every action I tried failed or did not bring the goal measurably closer…") |
| Recognises repeating / looping | "repeated actions refused" count; loops can't be committed (only measurable improvement is kept) |
| Recognises overspending | Per-task and per-room dollar budgets, the night's budget, the money chart; "budget" stops |
| Knows when to stop or ask for help | "Asked for help" table: each item says what it tried and why it stopped |
| Monitors its own progress | Progress is measured (cm of route blocked, contacts under an object), never self-reported |

---

## Honest answers to likely questions

- **"Is this a real robot?"** No. Everything happens in a simulated copy of the room built from a photo. A
  solved task is a plan to check in the real room, not a fact about it. We say so on every page.
- **"How accurate are the rooms?"** On 40 real rooms with ground truth: depth error 6 %, box centres off by
  0.31 m on average, about half of the objects found, and objects are boxes. The layout is realistic; the
  objects are simplified.
- **"Did it ever loop?"** It can't commit a loop: an action is kept only if it measurably improves the goal. What
  we do see is Claude proposing the same action again in the same situation. The supervisor refuses those
  (counted on the dashboard).
- **"Why Claude Haiku?"** It is the cheapest model, and it was good enough because physics checks every
  proposal. The model is one setting (`REALGROUNDS_MODEL=claude-sonnet-5-5`).
- **"What did the night cost?"** See the money tile. Haiku cost about $0.002 per room in our 10-room tests.
- **"Where do the masses come from?"** A catalog of typical values (marked "[typical]"), never measured.
- **"Whose rooms are these?"** NYU Depth V2 / SUN RGB-D: real rooms published for research. Our own photos stay
  local.

---

## The same moments in the app

| Moment | In the app |
|---|---|
| the night | tab 3 → the long run → "open the run's page" |
| a solution you can re-check | tab 3 → "Re-check every solution" (all of them, in seconds) |
| live, in this room | tab 1 → "Upload a room photo" (on a phone with `--share`: take the photo there) → "Ask Claude for task ideas" → "Run the task" |
| the stop rules | tab 2 → "Run with Claude" on each room |
| tests and results | tab 4 → "Run all tests" |

## What each command does

| Command | What it does |
|---|---|
| `python app.py [--share] [--cap 1]` | the app (everything below, in a browser) |
| `python demo.py check` | pre-stage check: key, models, Claude, physics, tests, a run to show |
| `python demo.py dashboard [run]` | builds and opens a run's page |
| `python demo.py replay [run] [room] [k]` | replays a solved task's kept actions on a fresh copy, with video, and checks the goal again |
| `python demo.py live PHOTO` | one photo → room → Claude writes 3 tasks → proofs with video → page |
| `python demo.py stops` | the test rooms with Claude: success / stuck / impossible, with video |
| `python night_shift.py --nyu 240 --split train+dev --types all --run long1` | the long run (resumes if interrupted) |
