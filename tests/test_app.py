"""The web app (app.py) end to end: a real server on a free port, the API a browser uses, a task run with the
fixed-rules planner (no API, no cost), and the safety checks (no files outside outputs/, no made-up room ids).

HOW TO RUN (from the realgrounds-voice/ folder)
    python -m unittest tests.test_app -v
"""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import app


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app.test_rooms()
        cls.lab = app.Lab(cap_usd=0.0)                      # no money: any Claude job must be refused
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(cls.lab))
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path: str):
        return json.load(urllib.request.urlopen(self.base + path))

    def post(self, path: str, body: dict):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req))

    def wait(self, job: str) -> dict:
        for _ in range(600):
            j = self.get(f"/api/job?id={job}")
            if j["status"] in ("done", "failed"):
                return j
            time.sleep(0.2)
        self.fail("job did not finish")

    def test_pages_rooms_and_a_free_task_run(self) -> None:
        self.assertEqual(urllib.request.urlopen(self.base + "/").status, 200)
        o = self.get("/api/overview")
        self.assertIn("test/light", [r["id"] for r in o["rooms"]])
        info = self.get("/api/room?id=test/light")
        self.assertIn("box #2", info["movable"])
        j = self.wait(self.post("/api/run", {"room": "test/light", "task": {"kind": "route"},
                                             "settings": {"planner": "rules", "video": False}})["job"])
        self.assertEqual(j["status"], "done")
        self.assertEqual(j["result"]["label"], "solvable")
        self.assertTrue(urllib.request.urlopen(self.base + j["result"]["transcript"]).read())

    def test_every_run_is_recorded_with_its_timeline(self) -> None:
        j = self.wait(self.post("/api/run", {"room": "test/heavy", "task": {"kind": "route"},
                                             "settings": {"planner": "rules", "video": False, "max_tries": 2}})["job"])
        r = j["result"]
        self.assertEqual((r["label"], r["outcome"]), ("unresolved", "budget"))
        self.assertEqual([s["type"] for s in r["steps"]][:1], ["plan"])
        self.assertIn("95 cm", r["task"]["start"])                         # the measured start, for the timeline
        recs = self.get("/api/experiments")["heavy"]["records"]
        self.assertEqual(recs[0]["time"], r["time"])                       # newest first: the replay pages find it

    def test_tasks_are_checked_before_any_proof(self) -> None:
        j = self.wait(self.post("/api/run", {"room": "test/light", "task": {"kind": "rests_on", "object": "desk #0",
                                                                             "target": "floor"}, "settings": {"planner": "rules"}})["job"])
        self.assertEqual(j["result"]["label"], "rejected")                 # fixed furniture

    def test_the_apps_money_cap_refuses_claude(self) -> None:
        j = self.wait(self.post("/api/run", {"room": "test/light", "task": {"kind": "route"},
                                             "settings": {"planner": "claude"}})["job"])
        self.assertEqual(j["status"], "failed")
        self.assertIn("budget", j["error"])

    def test_no_files_outside_outputs_and_no_made_up_rooms(self) -> None:
        for path in ("/files/../.env", "/files/%2e%2e/.env", "/web/../app.py"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(self.base + path)
            self.assertEqual(e.exception.code, 404, path)
        for rid in ("../../etc", "night/x/../../y", "upload/../../.env"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(self.base + "/api/room?id=" + urllib.request.quote(rid))
            self.assertEqual(e.exception.code, 400, rid)


if __name__ == "__main__":
    unittest.main()
