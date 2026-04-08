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
from config import DEVICE_TYPE, DEVICE_NAME, LOW_MEMORY_MODE, GPU_TOTAL_MEMORY_GB
from runtime_settings import build_subprocess_env, get_stored_groq_api_key, mask_secret, set_groq_api_key


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


def set_job_message(job: Job, message: str) -> None:
    with JOB_LOCK:
        job.message = message


def get_active_job() -> Job | None:
    with JOB_LOCK:
        for job in JOBS.values():
            if job.status in {"queued", "running"}:
                return job
    return None


def load_latest_report() -> dict:
    report_path = RESULTS_DIR / "report.json"
    if not report_path.exists():
        raise FileNotFoundError("results/report.json not found")
    return json.loads(report_path.read_text(encoding="utf-8"))


def get_settings_payload() -> dict:
    """Return frontend-safe runtime and settings status."""
    stored_key = get_stored_groq_api_key()
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    active_key = stored_key or env_key
    source = "backend" if stored_key else ("environment" if env_key else "missing")

    return {
        "groq": {
            "configured": bool(active_key),
            "masked_key": mask_secret(active_key),
            "source": source,
        },
        "runtime": {
            "device_type": DEVICE_TYPE,
            "device_name": DEVICE_NAME,
            "low_memory_mode": LOW_MEMORY_MODE,
            "memory_gb": round(GPU_TOTAL_MEMORY_GB, 1) if GPU_TOTAL_MEMORY_GB else None,
        },
    }


def run_pipeline(job_id: str) -> None:
    with JOB_LOCK:
        job = JOBS[job_id]
        job.status = "running"
        job.message = "Launching RESA_AI pipeline."

    command = [sys.executable, "-u", "main.py", job.input_path]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=build_subprocess_env(),
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
        elif current_phase is not None and clean and not clean.startswith(("=", "-")):
            set_job_message(job, clean)

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
        if path == "/api/settings":
            self.serve_json(get_settings_payload())
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
        if self.path == "/api/settings/groq":
            self.handle_groq_settings_update()
            return

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
            active_job = next((job for job in JOBS.values() if job.status in {"queued", "running"}), None)
            if active_job is not None:
                try:
                    os.remove(input_path)
                except OSError:
                    pass
                self.serve_json(
                    {
                        "error": "analysis_in_progress",
                        "message": f"Another paper is already being analyzed: {active_job.filename}",
                        "active_job_id": active_job.job_id,
                    },
                    status=HTTPStatus.CONFLICT,
                )
                return
            JOBS[job_id] = job

        threading.Thread(target=run_pipeline, args=(job_id,), daemon=True).start()
        self.serve_json({"job_id": job_id, "status": "queued"}, status=HTTPStatus.ACCEPTED)

    def handle_groq_settings_update(self) -> None:
        """Save or clear the Groq API key from the frontend."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected a valid JSON body")
            return

        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected a JSON object")
            return

        api_key = str(payload.get("groq_api_key", "") or "").strip()
        set_groq_api_key(api_key)

        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        else:
            os.environ.pop("GROQ_API_KEY", None)

        self.serve_json(get_settings_payload())

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
