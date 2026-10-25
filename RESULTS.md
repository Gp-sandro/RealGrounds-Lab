# RealGrounds Night Shift — tests and results

Every number here comes from a test or a run in this repository. The app's **Tests & results** tab shows the same
numbers live (`python app.py`). Everything the agent does happens in **simulation**: rooms are rebuilt from one
ordinary photo (sizes estimated, masses catalog values), and a "solved" task is a plan to check, not a fact about
the real room.

---

## 1. Automated tests: 44 of 44 pass

`python -m unittest discover -s tests -t .` (about 7 s; no network, no cost: the Claude tests use a fake API).

| Part | Tests | What they prove |
|---|---|---|
| Room understanding | 16 | what rests on / hangs on what, walls, overlapping boxes, records, physics settle / push / lift |
| The simulated body | 7 | commands end as expected; placements checked by contact; "too heavy" decided by physics; unknown verbs refused |
| Sandbox, goals, supervisor, loop | 7 | **a try leaves the world bit-identical**; a try predicts its commit; goal checks are deterministic; the supervisor refuses repeats, rejects loops, stops on no progress and on budgets; end to end: success / stuck / impossible |
| Claude planner, tasks, night supervisor | 9 | cost counted at list prices; refusals and cut-off answers stop the task; a planner that repeats itself is refused, then stopped; the money budget stops a task; bad task ideas are dropped before proof; reach uses the arm's reach; money, time and failure streaks stop the night |
| The web app | 5 | the API a browser uses; every run recorded with its timeline (what the experiment pages replay); a free task run; the app's money cap refuses Claude; no files outside `outputs/`, no made-up room ids |

---

## 2. Understanding a room from one photo

Measured in the research repo on **40 real rooms the system had never seen** (NYU Depth V2 / SUN RGB-D test
split, lens unknown), against human-labelled ground truth:

| | |
|---|---|
| depth error (AbsRel, lower is better) | 0.062 |
| rooms more than 15 % off in size | 0 % |
| 3D box centre error | 0.31 m |
| objects found and named right | 50 % |
| "rests on the floor" / "on another object" / "hangs on a wall" right | 92 % / 42 % / 62 % |
| time to build a room from one photo | about 1 min on an M5 MacBook (cached: seconds) |

## 3. Doing what it is told, checked by physics

| Room | Commands that ended as expected |
|---|---|
| real dorm room from photos, zone 1 | 13 / 14 |
| real dorm room from photos, zone 2 | 9 / 10 |
| hand-made test room | 13 / 13 |

---

## 4. The stop rules (hand-made test rooms, via the app)

The same task each time, "walk through the gap to the far side". Claude Haiku 5.5 as the planner, or the fixed
rules (no AI):

| Room | Planner | Outcome | Tries (undone) | Spend |
|---|---|---|---|---|
| light: a 1 kg box (catalog) and a chair block the gap | rules | **solved** (box moved onto the dresser) | 2 | $0 |
| light | Claude | **solved** | 1 | $0.0006 |
| heavy: a box set to 40 kg (test value); the arm lifts 15 kg | rules | **stopped: stuck**, asked for help | 5 | $0 |
| heavy | Claude | **stopped: stuck**, asked for help | 3 | $0.0008 |
| wall: fixed furniture blocks the gap | either | **impossible**, stopped at once; nothing to try | 0 | $0 |
| light, with a $0.0005 task budget | Claude | **stopped: budget** (money) before any try | 0 | $0.0007 |

---

## 5. Unattended runs on real rooms (NYU, one photo per room, Claude Haiku 5.5)

| Run | Rooms | Built / settle | Solved | Impossible | Stopped (stuck / budget) | Already true | Repeats refused | Bad ideas dropped | Help requests | Spend |
|---|---|---|---|---|---|---|---|---|---|---|
| test10 (first) | 10 | 10 / 9 | 5 | 3 * | 10 / 1 | 7 | 4 | 4 | 12 | $0.021 |
| test10b (after fixes) | 10 | 10 / 9 | 6 | 1 | 5 / 1 | 7 | 4 | 10 | 7 | $0.017 |
| long1 (in progress) | 240 planned | see the app | | | | | | | | |

\* One of test10's three "impossible" verdicts was wrong: "get next to a bottle on a deep sofa" counted only
floor within 0.42 m of the bottle's centre. With the fixed rule (the body's own arm reach: 0.75 m from the
object's outline) it is reachable. The other two (picture frames on walls behind furniture) are still impossible
under the fixed rule. Also since test10: task ideas the command layer would refuse anyway (e.g. lifting a
printer) are dropped before proof. Both fixes are in test10b and covered by tests.

**Cost and time:** about $0.002 and 1.4 min per new room (building from the photo takes about 1 min of that).

**Every solution can be re-checked.** Replaying only the kept actions on a fresh copy of the room (the physics is
deterministic): **6 of 6** test10b solutions hold again (app → Unattended runs → "Re-check every solution", or
`python demo.py replay`).

**Our own photo** (dorm, not in any dataset): built in about 1 min, 8 objects, settles. Claude wrote 3 tasks: 1 was
already true; 2 were stopped correctly (a backpack with no free floor in reach to put it on; a tote bag boxed in
on every side), each with a help request saying what it tried. No task was solved in that room.

---

## 6. What these results do not show

- **No real-world action or verification.** All results are in simulation.
- **Rooms are simplified:** about half the objects are found, objects are boxes, one photo shows part of a room.
- **The loop detector has not fired in a real run.** Loops can't be committed: an action is kept only if it
  measurably improves the goal. What does fire is the refusal of repeated proposals (4 per 10 rooms above).
- **Masses** are catalog typical values (marked "[typical]"), or test values (marked).
- **Claude's plans vary between runs.** The physics does not, which is why solutions are replayed to re-check them.
