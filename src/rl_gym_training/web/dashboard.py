"""Small browser dashboard using only the Python standard library."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rl_gym_training.sdk import RLGymTrainingSDK
from rl_gym_training.shared.paths import project_root, resolve_path

STATIC_DIR = Path(__file__).resolve().parent / "static"
COMMANDS = {
    "prepare": "prepare_data",
    "train-lstm": "train_world_model",
    "train-reinforce": "train_reinforce",
    "train-a2c": "train_a2c",
    "evaluate-random": "evaluate_random",
}


class DashboardHandler(BaseHTTPRequestHandler):
    sdk = RLGymTrainingSDK()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/app.css":
            self._send_file(STATIC_DIR / "app.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self._send_json(self._status_payload())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        prefix = "/api/run/"
        if not parsed.path.startswith(prefix):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        command = parsed.path.removeprefix(prefix)
        if command not in COMMANDS:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unknown command")
            return
        method = getattr(self.sdk, COMMANDS[command])
        try:
            result = method()
            payload = _json_ready(result)
            if command == "prepare":
                payload = {
                    "train_sequences": len(result.train),
                    "validation_sequences": len(result.validation),
                    "test_sequences": len(result.test),
                }
            elif command == "train-lstm":
                payload = {
                    "train_losses": result.train_losses,
                    "validation_loss": result.validation_loss,
                    "checkpoint_path": str(result.checkpoint_path),
                }
            self._send_json({"ok": True, "command": command, "result": payload})
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._send_json(
                {"ok": False, "command": command, "error": str(exc)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _status_payload(self) -> dict[str, object]:
        root = project_root()
        results = root / "results"
        artifacts = {
            name: (results / name).exists()
            for name in [
                "lstm_world_model.pt",
                "world_model_metrics.json",
                "reinforce_metrics.json",
                "a2c_metrics.json",
                "random_policy_metrics.json",
            ]
        }
        metrics = {}
        for file_name in [
            "world_model_metrics.json",
            "reinforce_metrics.json",
            "a2c_metrics.json",
            "random_policy_metrics.json",
        ]:
            path = results / file_name
            if path.exists():
                metrics[file_name] = json.loads(path.read_text(encoding="utf-8"))
        return {
            "project_root": str(root),
            "config": str(resolve_path("config/setup.yaml")),
            "artifacts": artifacts,
            "metrics": metrics,
        }


def serve_dashboard(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


def _json_ready(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "__dict__"):
            return {key: _json_ready(item) for key, item in value.__dict__.items()}
        return str(value)
