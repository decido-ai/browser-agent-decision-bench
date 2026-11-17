"""Checks that need no API key and no Mind2Web download: the published numbers re-score exactly, and `run` works
end to end against a local mock endpoint that speaks the /v1/systemone request shape."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ROOT / "data" / "steps-final.zip"
sys.path.insert(0, str(ROOT))
import decision_bench as B  # noqa: E402


def acc(name, split=None):
    steps = B.load_steps(STEPS)
    j = B.judge(steps, ROOT / "results" / name)
    us = [u for u, s in steps.items() if split is None or s["split"] in split]
    return round(100 * sum(j[u][0] for u in us) / len(us), 1)


def test_published_numbers_rescore_exactly():
    assert acc("decido-1-web-trained.jsonl") == 78.6
    assert acc("jev-1.14.jsonl") == 59.9
    assert acc("jev-1.13.jsonl") == 54.4
    assert acc("decido-1.jsonl") == 42.5
    assert acc("decido-1-web-trained.jsonl", {"test_website", "test_domain"}) == 78.2
    assert acc("jev-1.14.jsonl", {"test_website", "test_domain"}) == 58.6


class FirstOption(BaseHTTPRequestHandler):
    """Mock decision model: always picks the first option of every question."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answers = {}
        for qid, q in body["questions"].items():
            keys = list(q["criteria"])
            answers[qid] = {"choice": keys[0], "confidence": 1.0, "probabilities": {k: float(i == 0) for i, k in enumerate(keys)}}
        out = json.dumps({"model": "mock", "answers": answers}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_run_against_mock_endpoint(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), FirstOption)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "mock.jsonl"
    subprocess.run([sys.executable, str(ROOT / "decision_bench.py"), "run", "--steps", str(STEPS), "--limit", "5",
                    "--url", f"http://127.0.0.1:{srv.server_port}/v1/systemone", "--out", str(out)], check=True)
    srv.shutdown()
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 5 and all("error" not in r and r["model"] == "mock" and r["choice"] for r in recs)
