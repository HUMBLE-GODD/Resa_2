"""
Single-command launcher for the RESA_AI local app.

Usage:
    python start_resa.py
    python start_resa.py --no-browser
    python start_resa.py --restart
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from runtime_settings import build_subprocess_env


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
HEALTHCHECK_URL = f"{BASE_URL}/api/settings"
PID_FILE = ROOT / "data" / "frontend_server.pid"
LOG_OUT = ROOT / "frontend-server.out.log"
LOG_ERR = ROOT / "frontend-server.err.log"


def fetch_json(url: str, timeout: float = 2.0) -> dict | None:
    """Fetch JSON from a URL, returning None when unavailable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def is_server_ready() -> dict | None:
    """Return the settings payload when the frontend server is ready."""
    return fetch_json(HEALTHCHECK_URL, timeout=2.0)


def read_pid() -> int | None:
    """Read the stored frontend server PID, if present."""
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def write_pid(pid: int) -> None:
    """Persist the frontend server PID for future restarts."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid), encoding="utf-8")


def remove_pid_file() -> None:
    """Delete the saved PID file if it exists."""
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def is_process_running(pid: int) -> bool:
    """Return True if a process with this PID appears to still exist."""
    if pid <= 0:
        return False

    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_process(pid: int) -> None:
    """Terminate a known frontend server process."""
    if pid <= 0:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def find_frontend_server_pids() -> list[int]:
    """Find running frontend_server.py Python processes."""
    current_pid = os.getpid()

    if os.name == "nt":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'frontend_server.py' } | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pid = int(line)
                if pid != current_pid:
                    pids.append(pid)
        return pids

    result = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        command = parts[1]
        if pid != current_pid and "frontend_server.py" in command and "python" in command.lower():
            pids.append(pid)
    return pids


def stop_stale_server() -> None:
    """Stop a previously launched server when its PID is still recorded."""
    pid = read_pid()
    if pid is None:
        for extra_pid in find_frontend_server_pids():
            terminate_process(extra_pid)
        time.sleep(2)
        remove_pid_file()
        return

    pids_to_stop = {pid, *find_frontend_server_pids()}
    for target_pid in pids_to_stop:
        if is_process_running(target_pid):
            terminate_process(target_pid)

    time.sleep(2)
    remove_pid_file()


def launch_server() -> int:
    """Start the frontend server in the background and return its PID."""
    env = build_subprocess_env()
    env["PYTHONIOENCODING"] = "utf-8"

    creationflags = 0
    kwargs = {}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True

    with LOG_OUT.open("ab") as stdout_handle, LOG_ERR.open("ab") as stderr_handle:
        process = subprocess.Popen(
            [sys.executable, "-u", "frontend_server.py"],
            cwd=str(ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
            **kwargs,
        )

    write_pid(process.pid)
    return process.pid


def wait_for_server(timeout_seconds: int) -> dict | None:
    """Wait until the frontend server responds or the timeout expires."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = is_server_ready()
        if payload is not None:
            return payload
        time.sleep(1)
    return None


def format_runtime_summary(settings_payload: dict | None) -> str:
    """Build a short runtime summary for terminal output."""
    runtime = (settings_payload or {}).get("runtime", {})
    groq = (settings_payload or {}).get("groq", {})

    bits = []
    if runtime.get("device_name"):
        bits.append(runtime["device_name"])
    if runtime.get("device_type"):
        bits.append(runtime["device_type"])
    if runtime.get("memory_gb") is not None:
        bits.append(f"{runtime['memory_gb']:.1f} GB")
    if runtime.get("low_memory_mode"):
        bits.append("low-memory mode")
    if groq.get("configured"):
        bits.append(f"Groq {groq.get('source', 'configured')}")
    else:
        bits.append("Groq missing")

    return " | ".join(bits) if bits else "runtime status unavailable"


def parse_args() -> argparse.Namespace:
    """Parse launcher CLI arguments."""
    parser = argparse.ArgumentParser(description="One-command launcher for RESA_AI")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--restart", action="store_true", help="Restart the server even if it is already running")
    parser.add_argument("--timeout", type=int, default=45, help="Seconds to wait for the server to come online")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.restart:
        stop_stale_server()

    settings_payload = is_server_ready()
    if settings_payload is None:
        stop_stale_server()
        pid = launch_server()
        settings_payload = wait_for_server(args.timeout)
        if settings_payload is None:
            print("RESA_AI did not come online in time.")
            print(f"Check logs: {LOG_OUT.name}, {LOG_ERR.name}")
            return 1
        print(f"RESA_AI started at {BASE_URL} (PID {pid})")
    else:
        existing_pid = read_pid()
        if existing_pid and is_process_running(existing_pid):
            print(f"RESA_AI is already running at {BASE_URL} (PID {existing_pid})")
        else:
            print(f"RESA_AI is already running at {BASE_URL}")

    print(format_runtime_summary(settings_payload))

    if not args.no_browser:
        webbrowser.open(BASE_URL)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
