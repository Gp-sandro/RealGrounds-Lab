"""Export RealGrounds Lab as a read-only snapshot website (for publishing): the same pages, fed by JSON snapshots
of the API, with the recorded runs' videos. Nothing runs: buttons that need the backend are hidden.

USAGE (from the realgrounds-voice/ folder)
    python export_site.py [OUT_DIR]          # default outputs/site

Privacy: only hand-made test rooms and NYU Depth V2 rooms (research dataset) are exported. Uploaded photos, the
dorm rehearsal and anything else from people's own rooms are left out; the export stops if one slips in.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import app
from realgrounds.config import ROOT

PRIVATE = re.compile(r"upload|live-rehearsal|IMG_|dorm|smoke1|photo_\d", re.I)


def best_records(records: list[dict], n: int = 8) -> list[dict]:
    """One of each kind first (Claude / rules / 1-try limit / money budget), newest first, then the newest others."""
    seen, out = set(), []
    for r in records:
        key = (r.get("planner"), r["limits"]["max_tries"], r["limits"]["max_usd"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    out += [r for r in records if r not in out]
    return out[:n]


def main(argv: list[str]) -> int:
    out = Path(argv[0]) if argv else ROOT / "outputs" / "site"
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "rooms").mkdir(parents=True)

    # rooms: the three test rooms and the photo-built rooms of the latest complete NYU evaluation
    results = app.results()
    complete = next(r for r in results["runs"] if r["finished"] and r["rooms"] >= 5 and not PRIVATE.search(r["run"]))
    rooms = [r for r in app.list_rooms() if r["id"].startswith("test/") or r["id"].startswith(f"night/{complete['run']}/")]
    for r in rooms:
        info = app.room_info(r["id"])
        (out / "data" / "rooms" / (re.sub(r"[^A-Za-z0-9_-]", "_", r["id"]) + ".json")).write_text(json.dumps(info, default=str))

    overview = {"spent_usd": 0, "cap_usd": 0, "models": [], "rooms": rooms, "jobs": [], "queue": 0, "busy_runs": [],
                "snapshot_time": datetime.now().strftime("%Y-%m-%d %H:%M")}
    for run in results["runs"]:
        run["dashboard"] = None                                   # run pages are not part of the snapshot
    results["app_runs"] = [r for r in results["app_runs"] if not PRIVATE.search(r.get("room", ""))]
    experiments = app.experiments()
    for v in experiments.values():
        v["records"] = best_records(v["records"])

    texts = {"overview": json.dumps(overview, default=str), "results": json.dumps(results, default=str),
             "experiments": json.dumps(experiments, default=str)}
    texts.update({f"rooms/{p.stem}": p.read_text() for p in (out / "data" / "rooms").glob("*.json")})
    files = set()
    for name, text in texts.items():
        files |= set(re.findall(r'"/files/([^"]+)"', text))
        (out / "data" / f"{name}.json").write_text(text.replace('"/files/', '"files/'))
    bad = sorted(f for f in files if PRIVATE.search(f))
    if bad:
        raise SystemExit(f"refusing to export private files: {bad[:5]}")
    for f in sorted(files):
        src = app.OUT / f
        if src.is_file():
            (out / "files" / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, out / "files" / f)

    # the page: the app's own HTML, CSS and JS, inlined, in snapshot mode
    html = (app.WEB / "index.html").read_text()
    body = html.split("<body>", 1)[1].split("</body>", 1)[0]
    body = body.replace('<script src="/web/app.js"></script>', "")
    page = ("<title>RealGrounds Lab</title>\n"
            '<meta name="description" content="Real rooms from one photo, rebuilt in physics, where AI agents test actions '
            'on their own and know when to stop. A published snapshot.">\n'
            f"<style>\n{(app.WEB / 'style.css').read_text()}\n</style>\n{body}\n"
            f"<script>window.RG_STATIC = true;</script>\n<script>\n{(app.WEB / 'app.js').read_text()}\n</script>\n")
    (out / "index.html").write_text(page)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"{out}: {n} files, {size / 1e6:.1f} MB ({len(rooms)} rooms, {sum(len(v['records']) for v in experiments.values())} recorded runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
