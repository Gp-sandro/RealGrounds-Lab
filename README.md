# RealGrounds — a room an AI agent can act in, safely

Photos of a real room become a measured, physical copy of it (objects with sizes, positions, what rests on what,
in a MuJoCo physics engine). An AI agent gets tools on that copy - **look, try an action, commit it, check a
goal** - and works on its own until an independent supervisor stops it: goal reached, impossible, stuck, or out
of budget. Every try runs in an isolated snapshot and is undone; only actions that measurably help are committed.

Everything happens in the **simulated** room. A "simulated success" is a plan to check in the real room, not a
fact about it: sizes are estimated from photos and masses are catalog values.

Built for the c0mplied-16 hackathon, **Unsupervised** track. Forked from the RealGrounds research repo
(`../pixel/`), keeping only what works.

## What already works (measured)

On 40 real rooms the system had never seen (NYU Depth V2 / SUN RGB-D, lens unknown), from the research repo's
benchmark:

| | |
|---|---|
| depth error (AbsRel) | 0.062 |
| rooms more than 15 % off in size | 0 % |
| 3D box centre error | 0.31 m |
| things found and named right | 50 % |
| "rests on the floor" / "on another object" / "hangs on a wall" right | 92 % / 42 % / 62 % |
| time to build a room from one photo | ~1 min on an M5 MacBook (cached: ~5 s) |

The agent (a simulated body) does what it is told in rooms rebuilt from photos and checks every result in
physics: 13/14 and 9/10 commands on a real dorm room, 13/13 in a hand-made test room.

The autonomous loop, on hand-made test rooms (`python run_task.py --scene light|heavy|wall`):

| test room | what blocks the route | outcome | tries / commits |
|---|---|---|---|
| light | a box (catalog 1 kg) and a chair | **simulated success**: box onto the dresser | 2 / 1 |
| heavy | a box set to 40 kg (test value) | **stops (stuck)**, asks for help: too heavy for the 15 kg arm | 5 / 0 |
| wall | fixed furniture | **stops at once (infeasible)**, asks for help | 0 / 0 |

Every try restores the simulated world bit for bit (checked on each call), and every commit ends exactly as its
try did.

## Run it

```bash
cd realgrounds-voice
source ../.venv/bin/activate                                    # shared Python 3.14 venv
python build_room.py --input inputs/dorm                         # photos -> outputs/rooms/dorm/
python -m realgrounds.agent.run --scene outputs/rooms/dorm/zone_1 --tasks tasks_dorm_zone_1.yaml --no-video
python run_task.py --scene light --video                         # autonomous task -> outputs/tasks/light/
python night_shift.py --nyu 10 --run first                       # unattended: 10 real rooms -> dashboard.html
python app.py                                                    # RealGrounds Lab in the browser (or double-click
                                                                 #   "RealGrounds Lab.command"); --share for phones
python demo.py check                                             # the hackathon demo: see DEMO.md, RESULTS.md
python -m unittest discover -s tests -t .                        # 44 tests, ~7 s
```

Models (in `models/`, downloaded once, never at run time): `moge-2-vitl-normal` (MoGe-2, MIT, depth + lens),
`segformer-b0-ade` (floor / wall / ceiling), `grounding-dino-base` (objects by name), `sam-vit-base` (outlines).

## Layout

```
build_room.py          photos -> objects.json (the room), summary.json, room_mujoco.xml, floor plan
run_task.py            a goal -> the agent works on its own in simulation -> report, transcript, video
night_shift.py         photos -> rooms -> Claude writes tasks -> proofs; two supervisors; checkpoint/resume
demo.py                the hackathon demo, one command per moment (DEMO.md is the script)
app.py + web/          RealGrounds Lab: the whole project in a browser (no extra packages)
config.yaml            all settings (models, detector words, geometry and support rules)
realgrounds/
  depth.py             MoGe-2: metric depth + the camera lens from one photo
  segformer.py         floor / wall / ceiling pixels
  detect.py            Grounding DINO + SAM: objects by name with exact outlines (69 kinds in catalog.yaml)
  room.py              3D boxes, room planes, relations (supported_by, next_to, under, against, hanging_on)
  support.py           what rests on what: geometry x common sense learned from 6,397 real-room relations
  physics.py           MuJoCo export + checks (does the room settle? can the chair be pulled out?)
  model_cache.py       reuse the networks' outputs (--cache)
  agent/               a simulated body: floor path planner, pick up / put down / pull, plain-English commands
  sandbox.py           the agent's tools: observe, try_action (snapshot -> run -> restore), commit_action, check
  goals.py             objective goal checks: PathClear (route exists / how much is blocked / possible at all),
                       ObjectRestsOn (from the physics contacts)
  supervisor.py        limits, repeated actions and states, progress, the stop decision, the request for help
  autonomy.py          the loop: measure -> propose -> try -> commit the best; rule_based_planner (no model)
  claude_planner.py    Claude as the planner (structured output, spend counted at list prices)
  tasks.py             tasks for a room (automatic route + Claude-written) and their proof
  run_supervisor.py    the night's supervisor: money, time, failure streaks, help queue, timeline
  dashboard.py         the night as one page (dashboard.html)
tests/                 room relations, physics, agent, sandbox / goals / supervisor / autonomous loop
```

## Next

1. A real photographed room with a blocked route (the photo pipeline -> the same loop).
2. Claude as the planner (same `planner(env, goal)` signature as `rule_based_planner`; the supervisor stays in
   charge of every try, commit and stop).
3. An MCP server exposing observe / try_action / commit_action / check, once the loop with Claude works.

## Honest limits

One photo covers one part of a room (photos that don't overlap become separate zones); small things are found
about half the time; objects are boxes; a guessed answer is never given as fact - the record says what was seen
and what was inferred. The agent's body is simulated: nothing here moves anything in a real room, and a simulated
result is not real-world verification.
