"""Run the complete EndoPolyp-Seg pipeline with one IDE action."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_NAME = "EndoPolyp-Seg"
DEMO_URL = "http://127.0.0.1:7895"
LOG_PATH = ROOT / "run_project.log"


def log(message: str) -> None:
    """Show progress in the IDE and keep a copy when the IDE hides its console."""
    print(message, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        # Logging must never prevent the actual pipeline from running.
        pass


def run_stage(label: str, script: str, *args: str) -> None:
    log(f"\n=== {label} ===")
    subprocess.run([sys.executable, "-u", script, *args], cwd=ROOT, check=True)


def report_matches(path: Path, **expected: object) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(report.get(key) == value for key, value in expected.items())


def demo_is_running() -> bool:
    """Return true only when the port serves this Gradio app, not any HTTP server."""
    try:
        with urllib.request.urlopen(DEMO_URL, timeout=1.5) as response:
            response_body = response.read(20_000).decode("utf-8", errors="ignore")
        with urllib.request.urlopen(f"{DEMO_URL}/config", timeout=1.5) as config:
            config_body = config.read(50_000).decode("utf-8", errors="ignore")
        return (
            response.status == 200
            and PROJECT_NAME in response_body
            and config.status == 200
            and '"components"' in config_body
            and PROJECT_NAME in config_body
        )
    except (OSError, urllib.error.URLError):
        return False


def open_demo() -> None:
    """Open the local page and always print a usable fallback URL."""
    try:
        if sys.platform == "win32":
            try:
                os.startfile(DEMO_URL)
                opened = True
            except OSError:
                opened = webbrowser.open_new_tab(DEMO_URL)
        else:
            opened = webbrowser.open_new_tab(DEMO_URL)
    except Exception as exc:  # pragma: no cover - browser behavior is OS-specific
        opened = False
        log(f"Browser open request failed: {exc}")
    log(f"Demo URL: {DEMO_URL}")
    log(f"Browser open request: {'accepted' if opened else 'not accepted'}; paste the URL above if needed.")


def keep_runner_visible() -> None:
    log("Demo is ready. Keep this Run window open; stop it with the red square in the IDE.")
    try:
        while demo_is_running():
            time.sleep(2)
    except KeyboardInterrupt:
        log("Runner stopped. The existing demo service is unchanged.")


def main() -> None:
    manifest = ROOT / "data" / "processed" / "manifest.csv"
    best_weights = ROOT / "artifacts" / "best.pt"
    smoke_weights = ROOT / "artifacts" / "smoke.pt"
    best_onnx = ROOT / "artifacts" / "best.onnx"

    log(f"{PROJECT_NAME} one-click runner started.")
    log(f"Project directory: {PROJECT_NAME}")

    if not manifest.is_file():
        run_stage("Prepare data", "prepare_data.py")
    else:
        log("=== Prepare data: existing manifest found, skipped ===")

    if not best_weights.is_file():
        if not smoke_weights.is_file():
            run_stage("Smoke training", "train.py", "--smoke")
        run_stage("Formal training", "train.py")
    else:
        log("=== Training: existing best.pt found, skipped ===")

    if not report_matches(ROOT / "reports" / "classical_baseline.json", split="test"):
        run_stage("OpenCV baseline", "classical_baseline.py")
    if not report_matches(ROOT / "reports" / "evaluation.json", split="test", weights="artifacts/best.pt"):
        run_stage("Test evaluation", "evaluate.py", "--overwrite")
    if not best_onnx.is_file() or not (ROOT / "reports" / "onnx_validation.json").is_file():
        run_stage("ONNX export", "export_onnx.py")
    if not report_matches(ROOT / "reports" / "onnx_benchmark.json", model="artifacts/best.onnx", images=50):
        run_stage("CPU benchmark", "benchmark.py")

    if demo_is_running():
        log(f"=== Gradio demo: an existing {PROJECT_NAME} service is already running ===")
        open_demo()
        keep_runner_visible()
    else:
        log("=== Gradio demo: starting app.py ===")
        log(f"When the browser does not open automatically, visit {DEMO_URL}")
        demo_process = subprocess.Popen([sys.executable, "-u", "app.py"], cwd=ROOT)
        for _ in range(120):
            if demo_is_running():
                open_demo()
                break
            if demo_process.poll() is not None:
                raise subprocess.CalledProcessError(demo_process.returncode, [sys.executable, "-u", "app.py"])
            time.sleep(0.5)
        else:
            demo_process.terminate()
            raise TimeoutError(f"Gradio did not start within 60 seconds. Expected {DEMO_URL}")
        try:
            demo_process.wait()
        except KeyboardInterrupt:
            demo_process.terminate()
            log("Demo stopped.")


if __name__ == "__main__":
    main()
