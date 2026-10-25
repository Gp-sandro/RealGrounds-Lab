# RealGrounds Night Shift — design

**One line:** an AI agent that works unattended overnight turning ordinary photos of real rooms into
**physics-checked simulation rooms with tasks proven solvable (or proven impossible)**, and that knows when it is
stuck, looping, overspending, or needs a person.

Track: **3 — Unsupervised**. Status: design. What exists today is marked ✅; what must be built is marked 🔨.

---

## 1. The problem and who has it

People who build and test AI that acts in the physical world (robot learning, embodied AI, world models, agent
developers) need **many realistic rooms to train and test in**. Today they choose between:

- **synthetic rooms** (procedurally generated or written by a language model): unlimited, but the layouts are not
  real;
- **3D scans** (Matterport-style scanners): real, but every room needs scanning hardware and a visit.

Ordinary photos of real rooms exist in huge numbers. Turning them into simulation rooms is possible (RealGrounds
does it in about a minute), but **every built room has errors** (an object floating, two objects overlapping, a
missed support), and a task written for a room may not be solvable in it. Someone has to check every room and
every task by hand. That checking is what doesn't scale, and it is exactly the kind of long, repetitive,
judgement-heavy work an unsupervised agent should do.

**First user: us.** RealGrounds is a world-model research project; it needs this library itself. Other users
(robot-learning and agent-evaluation people) need the same thing. A task with **proof** that it is solvable,
or impossible, is what an agent benchmark needs.

## 2. What it does

You give it a folder of room photos and a budget, and go to sleep. For every photo:

| Step | Who does it | What happens | Exists? |
|---|---|---|---|
| 1. Build | pipeline | photo → measured room (objects, sizes, what rests on what) → physics | ✅ `build_room.py` |
| 2. Inspect | pipeline | objective checks: does the room settle, do objects overlap at the start, can the body stand, how far did each object drift | ✅ `physics.physics_proofs` |
| 3. Repair | **Claude** proposes, the sandbox tries, the supervisor decides | reads the failed checks, picks repairs from a fixed menu (rebuild with other settings, drop a phantom object, rest an object on its support, make an object fixed), tries each on a copy of the room, and keeps a repair only if the checks measurably improve | 🔨 the try/commit/supervisor pattern ✅, applied to the room files 🔨 |
| 4. Write tasks | **Claude** | proposes tasks that fit this room ("put the book on the shelf", "clear a route from the door to the desk", "pull out the chair") | 🔨 |
| 5. Prove tasks | **Claude** plans, the sandbox tries, the supervisor decides | solves each task in simulation; labels it **solvable** (with the solution), **impossible** (with the measured reason), or **unresolved** (budget or stuck) | ✅ for route tasks (`autonomy.run`); 🔨 other goals, Claude planner |
| 6. Publish | pipeline | a room card: what is estimated vs seen, the quality checks, the tasks with their evidence | 🔨 |

In the morning: a **library** of rooms and proven tasks, a **report** of the night, and a short **queue for a
person**, listing only the rooms and tasks the agent couldn't settle, each with what it tried.

The library is then **used through MCP**: any agent (Claude Desktop, Claude Code, someone's own agent) can list
rooms, load one, observe it, try and commit actions, and check goals, with the same supervisor inside the server.

## 3. Why it fits Track 3: every requirement arises from real work

| Track 3 requires | Where it happens naturally |
|---|---|
| Operates for hours unsupervised | Hundreds of photos × (build ~1 min + repair + several tasks to prove). The hours are the job, not padding. |
| Recognises being stuck | A room whose checks don't improve after N repairs; a task where no tried action helps (too heavy, nothing graspable) |
| Recognises looping | Repairs that undo each other (lifting an object makes it float, lowering it makes it overlap); a chair moved to clear one route that blocks another and gets moved back. Detected from the room's measured state, not the agent's opinion. |
| Recognises overspending | Spend in dollars, tokens and time, per room and per night; cost per proven task tracked; a room whose cost rises without progress is abandoned and queued for a person |
| Stops or asks for help | Per room or task: flagged with evidence. Per night: e.g. 5 builds failing in a row, or the success rate collapsing, means something is wrong with the run itself, so stop everything and report. |
| Monitors its own progress | Every step is scored by measurement: checks passed, cm of route blocked, tasks proven. A live timeline shows rooms per hour, cost per proven task, and each stop. |

**The AI is doing real work.** Claude diagnoses broken rooms, chooses repairs, writes tasks and plans
solutions: the judgement calls. The pipeline does the measuring, and the supervisor does the deciding to stop.
Every Claude decision is logged with its measured outcome.

## 4. Why it is unique

1. **From ordinary photos to physics.** No scanner, no synthetic layout: any photographed room can become a
   simulation room.
2. **Try before commit, everywhere.** The same mechanism checks a repair to the room and an action in the room:
   run on a snapshot, measure, undo (verified bit for bit), keep only what measurably helps.
3. **Progress is measured, not claimed.** The agent never decides it succeeded; physics and geometry do. This is
   what makes leaving it alone for a night safe.
4. **Honest labels.** Every room says what was seen, inferred or estimated; every result is "simulated", never a
   claim about the real room.

## 5. The program

```
night_shift.py                 the run: photos folder + budget -> library + report (checkpoint/resume)  🔨
realgrounds/
  sandbox.py        ✅          observe / try_action / commit_action / check (in a room)
  goals.py          ✅ + 🔨     PathClear, ObjectRestsOn  + goals for other task kinds
  supervisor.py     ✅ + 🔨     per-task limits  + spend in $ / tokens / time
  autonomy.py       ✅ + 🔨     the try/commit loop  + Claude as planner (same interface)
  repair.py         🔨          room checks -> repair menu -> try on a copy -> keep if better
  tasks.py          🔨          Claude writes tasks for a room; each is proven with autonomy.run
  run_supervisor.py 🔨          the night: budgets, failure-rate alarms, help queue, live timeline
  mcp_server.py     🔨          the library as tools for any agent, supervisor inside
outputs/library/<room>/        room files, room card, tasks with proof
outputs/night/<run>/           report.md, timeline.jsonl, help_queue.md, spend.json
```

## 6. The demo (for judges)

1. **The morning report** of a real overnight run, with timestamps on every event: rooms processed, repaired,
   flagged; tasks proven solvable / impossible; spend; every time it caught itself (stuck, loop, overspend), with
   the evidence.
2. **Live:** a photo of the room we are standing in → built in a minute → the agent inspects, repairs, writes
   three tasks and proves them. One is proven impossible and it says why.
3. **Use:** Claude Desktop, through MCP, loads that room and attempts a task; the server's supervisor stops it
   when it repeats itself.

## 7. Build order (fastest path to something that works end to end)

1. Claude as the planner for the existing route tasks: the first real AI decisions, with spend recorded.
2. `night_shift.py` over a small batch (10 NYU rooms): build → inspect → prove route tasks; checkpoint/resume;
   run-level supervisor. **Measure** minutes and dollars per room here, before promising hours.
3. `repair.py`: the repair menu, tried on copies.
4. `tasks.py`: Claude writes tasks beyond routes; more goal kinds.
5. `mcp_server.py`.
6. The overnight run, the live demo, the pitch.

After step 2 there is a working end-to-end demo; each later step makes it stronger, and nothing later is
needed for it to work.

## 8. Risks — said plainly

- **Room quality.** About half the objects are found, objects are boxes, and one photo shows part of a room. The
  rooms are realistic in **layout**, simplified in **objects**. The library must say so.
- **Repairs may not fix much.** If most broken rooms can't be repaired, the honest result is a well-reasoned
  help queue. That is still correct behaviour, but a weaker demo. Step 3 will tell.
- **Cost.** Claude runs all night; the budget is set before the run and enforced by the supervisor. The actual
  cost per room is unknown until step 2.
- **Data licence.** NYU / SUN RGB-D are for research and evaluation; a library built from them is for research
  use. Rooms from our own photos have no such limit (and stay private unless their owners agree).
- **"Will it be used?"** Certain by us. By others: plausible (they need rooms and proven tasks), not proven. A
  public sample library after the hackathon would test it.
