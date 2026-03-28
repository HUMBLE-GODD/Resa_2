"""
Lightweight server for the RESA_AI frontend.

Usage:
    python frontend_server.py
"""
from __future__ import annotations

import cgi
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
RESULTS_DIR = ROOT / "results"
UPLOADS_DIR = ROOT / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PHASES = [
    ("ingestion", "Data ingestion", "Parse PDF pages, structure, sections, equations, and references."),
    ("preprocessing", "Preprocessing", "Clean text, preserve math notation, and build dual representations."),
    ("features", "Feature engineering", "Extract keywords, entities, and mathematical structures."),
    ("models", "Transformer models", "Generate summaries, classify sections, and compute similarity."),
    ("analysis", "Advanced NLP analysis", "Discover topics, build semantic search, and classify equations."),
    ("groq", "LLM insights", "Produce summary, ELI5, contributions, applications, and limitations."),
    ("output", "Report generation", "Assemble JSON, markdown report, and visual outputs."),
]


@dataclass
class Job:
    job_id: str
    filename: str
    input_path: str
    status: str = "queued"
    message: str = "Queued for analysis."
    created_at: float = field(default_factory=time.time)
    phases: list[dict] = field(default_factory=list)
    report: dict | None = None
    error: str | None = None


JOBS: dict[str, Job] = {}
JOB_LOCK = threading.Lock()


def make_phase_list() -> list[dict]:
    return [
        {"id": phase_id, "title": title, "detail": detail, "status": "idle"}
        for phase_id, title, detail in PIPELINE_PHASES
    ]


def update_phase(job: Job, phase_index: int, status: str, message: str) -> None:
    with JOB_LOCK:
        for index, phase in enumerate(job.phases):
            if index < phase_index and phase["status"] != "completed":
                phase["status"] = "completed"
            elif index == phase_index:
                phase["status"] = status
        job.message = message
        if status == "running":
            job.status = "running"
        elif status == "failed":
            job.status = "failed"


def load_latest_report() -> dict:
    report_path = RESULTS_DIR / "report.json"
    if not report_path.exists():
        raise FileNotFoundError("results/report.json not found")
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_pipeline(job_id: str) -> None:
    with JOB_LOCK:
        job = JOBS[job_id]
        job.status = "running"
        job.message = "Launching RESA_AI pipeline."

    command = [sys.executable, "main.py", job.input_path]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    phase_lookup = {
        "DATA INGESTION": 0,
        "PREPROCESSING": 1,
        "FEATURE ENGINEERING": 2,
        "TRANSFORMER MODELS": 3,
        "ADVANCED NLP ANALYSIS": 4,
        "GROQ LLM ANALYSIS": 5,
        "OUTPUT GENERATION": 6,
    }
    current_phase = None

    for line in iter(process.stdout.readline, ""):
        clean = line.strip()
        if clean.startswith("PHASE ") and ":" in clean:
            phase_name = clean.split(":", 1)[1].strip()
            current_phase = phase_lookup.get(phase_name)
            if current_phase is not None:
                update_phase(job, current_phase, "running", f"{phase_name.title()} in progress.")
        elif clean.startswith("[OK] Phase") and current_phase is not None:
            update_phase(job, current_phase, "completed", f"{job.phases[current_phase]['title']} completed.")
        elif clean.startswith("[FAIL] Phase") and current_phase is not None:
            update_phase(job, current_phase, "failed", clean)

    return_code = process.wait()
    with JOB_LOCK:
        if return_code == 0:
            for phase in job.phases:
                if phase["status"] == "running":
                    phase["status"] = "completed"
            job.status = "completed"
            job.message = "Analysis complete."
            job.report = load_latest_report()
        else:
            if current_phase is not None and job.phases[current_phase]["status"] != "failed":
                job.phases[current_phase]["status"] = "failed"
            job.status = "failed"
            job.error = f"Pipeline exited with code {return_code}."
            job.message = job.error


class ResaHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self.serve_file(FRONTEND_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self.serve_file(FRONTEND_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self.serve_file(FRONTEND_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        if path.startswith("/frontend/"):
            self.serve_static(FRONTEND_DIR, path[len("/frontend/"):])
            return
        if path.startswith("/results/"):
            self.serve_static(RESULTS_DIR, path[len("/results/"):])
            return
        if path == "/api/report/latest":
            try:
                self.serve_json(load_latest_report())
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "No report available yet")
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOB_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    self.send_error(HTTPStatus.NOT_FOUND, "Job not found")
                    return
                self.serve_json(job.__dict__)
                return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        ctype, pdict = cgi.parse_header(self.headers.get("Content-Type"))
        if ctype != "multipart/form-data":
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data")
            return

        pdict["boundary"] = pdict["boundary"].encode("utf-8")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "paper" not in form:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing paper file")
            return

        upload = form["paper"]
        filename = os.path.basename(upload.filename or "paper.pdf")
        job_id = uuid.uuid4().hex[:12]
        input_path = UPLOADS_DIR / f"{job_id}-{filename}"
        with open(input_path, "wb") as target:
            shutil.copyfileobj(upload.file, target)

        job = Job(job_id=job_id, filename=filename, input_path=str(input_path), phases=make_phase_list())
        with JOB_LOCK:
            JOBS[job_id] = job

        threading.Thread(target=run_pipeline, args=(job_id,), daemon=True).start()
        self.serve_json({"job_id": job_id, "status": "queued"}, status=HTTPStatus.ACCEPTED)

    def serve_static(self, root: Path, relative_path: str) -> None:
        safe_path = (root / relative_path).resolve()
        if root not in safe_path.parents and safe_path != root:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not safe_path.exists() or not safe_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self.serve_file(safe_path, self.guess_type(safe_path.suffix))

    def serve_file(self, file_path: Path, content_type: str) -> None:
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def serve_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def guess_type(extension: str) -> str:
        return {
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".html": "text/html; charset=utf-8",
        }.get(extension.lower(), "application/octet-stream")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), ResaHandler)
    print("RESA_AI frontend running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
