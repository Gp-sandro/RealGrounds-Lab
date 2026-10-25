# RealGrounds Voice — guide for coding agents

Read README.md first. This is the hackathon fork of RealGrounds (research repo: `../pixel/`, see its Agent.md);
only code that works and is measured was copied here.

## Rules

- **Simulated is not real**: every agent action runs in the simulated room. Never call a simulated result a
  real-world outcome; label it (`SIMULATED` in `realgrounds/sandbox.py`). Sizes are estimates and masses are
  catalog values ("[typical]") or marked test values - never present them as measured.
- **The supervisor decides when to stop**, not the planner: every try and commit goes through
  `realgrounds/supervisor.py`.
- **Answer from the measured room, never guess**: object records say what was seen vs inferred; when the room
  can't tell, the answer must say so ("I can't see what the fan stands on").
- **Measure before claiming**: run `python -m unittest discover -s tests -t .` before and after changes; rebuild
  `inputs/dorm` (`python build_room.py --input inputs/dorm --cache`) and check both zones still settle in physics.
- Models load from `models/` only; no downloads at run time. Caches stay in `.cache/`.
- Photos of people's rooms are private: git-ignored, never published.
- Python 3.14, shared venv `../.venv`. Record new dependencies in requirements.txt.

## Where things are

| | |
|---|---|
| photos -> room | `build_room.py` (`--cache` reuses network outputs: seconds instead of a minute) |
| settings | `config.yaml`; object kinds and their physical attributes: `realgrounds/catalog.yaml` |
| what rests on what | `realgrounds/support.py` + `support_prior.yaml` (learned in the research repo from NYU's train split) |
| physics | `realgrounds/physics.py` (MuJoCo) |
| agent | `realgrounds/agent/` (`python -m realgrounds.agent.run --scene <room folder> --tasks <file>`) |
| agent tools | `realgrounds/sandbox.py` (observe / try_action / commit_action / check), `goals.py` |
| autonomous loop | `realgrounds/autonomy.py` + `supervisor.py`; `python run_task.py --scene light` |

The benchmark (dataset, scorer, depth comparisons) stays in the research repo: `../pixel/lab/eval_rooms.py`,
`score_rooms.py`, `datasets/`.
