# RealGrounds — what we have (October 2026)

A working base to reshape into a hackathon entry. Everything below runs today and was measured; anything that
is a plan or an estimate is marked as such.

## The idea in one line

Photos of a real room become a **measured, physical copy of it** — every object with its size, position and what
it rests on, loaded in a physics engine — so an AI can reason about the room and test actions in it before
anything happens in the real room.

## 1. Photos → measured room

`python build_room.py --input <folder of photos>` (about 1 min per photo on an M5 MacBook; ~5 s when cached).

| Step | Model / method | Gives |
|---|---|---|
| Depth + lens | MoGe-2 | distance per pixel in metres, the camera's field of view (no calibration needed) |
| Room structure | SegFormer | floor, walls, ceiling |
| Objects | Grounding DINO + SAM | objects by name with exact outlines (69 kinds in the catalog) |
| 3D | geometry | a 3D box per object, room planes, camera height |
| Relations | geometry + a prior learned from 6,397 labelled real-room relations | rests on, hangs on, next to, under, against |
| Physics | MuJoCo | the room loads and settles; checks such as "can the chair be pulled out" |

Each object gets a record: name, box, position ("on desk surface"), material, state, affordances (grasp, lift,
open, pull…), mass, and **which of these were seen vs inferred**.

**Measured accuracy** on 40 real rooms it had never seen (NYU Depth V2 / SUN RGB-D, lens unknown):

| | |
|---|---|
| depth error (AbsRel) | 0.062 |
| rooms more than 15 % off in size | 0 % |
| 3D box centre error | 0.31 m |
| objects found and named right | 50 % |
| "rests on the floor" / "on another object" / "hangs on a wall" right | 92 % / 42 % / 62 % |

## 2. A simulated body that does what it is told

`realgrounds/agent/` — plain-English commands ("pick up the book and put it on the nightstand") are parsed,
resolved to objects ("it", "the mug on the desk", synonyms), checked against the objects' attributes, and run by
a simulated body in physics. The body is 0.4 m wide, steps over things lower than 20 cm, plans walking routes on
the floor, and its arm lifts at most 15 kg. Verbs: walk, pick up, put on, pull out, push, drop.

Physics checks every result; impossible requests are refused with a reason ("desk #0 is fixed furniture"), and
failures are measured ("too heavy: rose 0 cm").

| Room | Commands ending as expected |
|---|---|
| real dorm room (from photos), zone 1 | 13 / 14 |
| real dorm room (from photos), zone 2 | 9 / 10 |
| hand-made test room | 13 / 13 |

## 3. Tools for an autonomous agent (new)

| Tool | What it does | Verified |
|---|---|---|
| `observe()` | the room as data, each property with its source (estimated / catalog / test value) | — |
| `try_action(text)` | runs an action from a snapshot, measures it, restores the snapshot | world bit-identical afterwards, checked on every call (also on the photo-built dorm: 24 tries) |
| `commit_action(text)` | runs the action on the persistent **simulated** room | ends exactly as its try did |
| `check(goal)` | objective goal checks from geometry and collisions: is there a walking route, how many cm are blocked, by what, is it possible at all; does X rest on Y | deterministic |

**Supervisor** (separate from whatever plans the actions): limits on tries, commits and simulated time; refuses a
repeated action in the same state; stops on a loop (back to an earlier state); commits only actions that
measurably help; stops when stuck, infeasible or out of budget, and asks for help with what it tried.

**Demo today** (`python run_task.py --scene light|heavy|wall --video`; report, transcript, log, stills, video):

| Test room | Outcome | Tries / commits |
|---|---|---|
| light: a box and a chair block a route | simulated success (box onto the dresser) | 2 / 1 |
| heavy: the box set to 40 kg (test value) | stops (stuck), asks for help: too heavy | 5 / 0 |
| wall: fixed furniture blocks the route | stops at once (infeasible), asks for help | 0 / 0 |

30 automated tests pass (`python -m unittest discover -s tests -t .`, ~4 s).

## 4. Assets

- **Data:** NYU Depth V2 + SUN RGB-D in `../pixel/datasets/` — 1,449 photos of 464 real rooms with ground truth
  (depth, outlines, support, 3D boxes). Licence: research and evaluation use.
- **Models:** all local (`models/`), no downloads at run time.
- **Own photos:** a dorm room (two zones), private and git-ignored.

## 5. Not there yet

- **No AI planner yet.** The planner is a rule list; Claude is meant to replace it (same interface).
- **Short runs.** One task takes seconds; nothing long-running exists yet.
- **No real-world action or check.** Everything happens in the simulated room. A simulated success is a plan to
  check, not a fact.
- **Estimates, not measurements:** sizes come from photos (box centres off by 0.31 m on average); masses are
  catalog typical values.
- **Partial rooms and missed objects:** one photo shows part of a room; half of the objects are found; objects
  are boxes.
- **A snapshot:** the room is as it was when photographed.
- **No MCP server yet**, no voice, no live data.

## 6. How it maps to the tracks

| Track | What of ours it would use | What it would still need | Fit |
|---|---|---|---|
| 3. Unsupervised | the physics copy as a place to test actions; the supervisor (stuck, loop, budget, stop/help) | Claude as the planner; a job that takes hours (e.g. many rooms end to end); real spend tracking (tokens, $, time); checkpoint/resume; optionally an MCP server with the supervisor inside | best: the core idea is what the track asks for |
| 2. No App | the measured room: where things are, is a path clear | voice in and out; screenless photo capture; a way around the room being a snapshot | medium |
| 4. 1 Billion People | the measured room: will this fit, home safety | much higher detection reliability; a clear edge over Apple RoomPlan / IKEA Kreativ | weak |
| 1. Dispatch | little | live data and real things moving | poor |
