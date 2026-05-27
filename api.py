"""HTTP API and static frontend server for emotion-recognition inference."""

from __future__ import annotations

import argparse
import csv
import mimetypes
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import urlparse
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import EmotionRecognizer


recognizer = EmotionRecognizer(PROJECT_ROOT)
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESULTS_DIR = PROJECT_ROOT / "Results"
UPLOAD_DIR = PROJECT_ROOT / "uploads"


class EmotionAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing speech, text, and fusion predictions."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok", "models": self._model_status()})
            return
        if parsed.path == "/api/results":
            self._send_json(self._results_payload())
            return
        if parsed.path == "/api/sample":
            self._send_json(self._sample_payload())
            return
        if parsed.path.startswith("/plots/"):
            self._send_file(RESULTS_DIR / parsed.path.lstrip("/"))
            return
        self._serve_frontend(parsed.path)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/predict/upload":
                result = self._predict_upload()
            else:
                payload = self._read_json()
            if parsed.path == "/predict/speech":
                result = recognizer.predict_speech(payload["speech_path"])
            elif parsed.path == "/predict/text":
                result = recognizer.predict_text(payload["text"])
            elif parsed.path == "/predict/fusion":
                result = recognizer.predict_fusion(payload["speech_path"], payload["text"])
            elif parsed.path == "/api/predict/text":
                result = recognizer.predict_text(payload["text"])
            elif parsed.path == "/api/predict/upload":
                pass
            else:
                self._send_json({"error": "Not found"}, status=404)
                return
            self._send_json(result)
        except KeyError as exc:
            self._send_json({"error": f"Missing field: {exc.args[0]}"}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw_body or "{}")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "File not found"}, status=404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self._send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_frontend(self, path: str) -> None:
        target = "index.html" if path in {"", "/"} else path.lstrip("/")
        file_path = (FRONTEND_DIR / target).resolve()
        if not str(file_path).startswith(str(FRONTEND_DIR.resolve())):
            self._send_json({"error": "Invalid path"}, status=400)
            return
        if not file_path.exists() or not file_path.is_file():
            file_path = FRONTEND_DIR / "index.html"
        self._send_file(file_path)

    def _model_status(self) -> dict:
        return {
            "speech": (PROJECT_ROOT / "models" / "speech_pipeline" / "speech_model.npz").exists(),
            "text": (PROJECT_ROOT / "models" / "text_pipeline" / "text_model.json").exists(),
            "fusion": (PROJECT_ROOT / "models" / "fusion_pipeline" / "fusion_model.npz").exists(),
        }

    def _read_csv_rows(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _results_payload(self) -> dict:
        return {
            "accuracy": self._read_csv_rows(RESULTS_DIR / "all_model_accuracy_table.csv"),
            "speech_report": self._read_csv_rows(
                RESULTS_DIR / "speech_test_classification_report.csv"
            ),
            "text_report": self._read_csv_rows(
                RESULTS_DIR / "text_test_classification_report.csv"
            ),
            "fusion_report": self._read_csv_rows(
                RESULTS_DIR / "fusion_test_classification_report.csv"
            ),
            "plots": {
                "speech": "/plots/speech_temporal_representation.svg",
                "text": "/plots/text_contextual_representation.svg",
                "fusion": "/plots/fusion_representation.svg",
            },
        }

    def _sample_payload(self) -> dict:
        rows = self._read_csv_rows(PROJECT_ROOT / "data" / "metadata.csv")
        for row in rows:
            if row.get("split") == "test":
                return {
                    "speech_path": row["speech_path"],
                    "text": row["text"],
                    "emotion": row["emotion"],
                }
        return {}

    def _predict_upload(self) -> dict:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart/form-data request.")
        boundary_token = "boundary="
        if boundary_token not in content_type:
            raise ValueError("Missing multipart boundary.")

        boundary = content_type.split(boundary_token, 1)[1].strip().strip('"')
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        fields, files = parse_multipart(body, boundary.encode("utf-8"))
        text = fields.get("text", "").strip()
        mode = fields.get("mode", "all").strip() or "all"

        result: dict[str, object] = {"text": text, "mode": mode}
        if mode in {"text", "all"}:
            if not text:
                raise ValueError("Text is required for text prediction.")
            result["text_prediction"] = recognizer.predict_text(text)

        if mode in {"speech", "fusion", "all"}:
            uploaded = files.get("audio")
            if uploaded is None:
                raise ValueError("Audio file is required for speech or fusion prediction.")
            if not uploaded["filename"].lower().endswith((".wav", ".aiff", ".aif")):
                raise ValueError("Please upload a WAV or AIFF audio file.")
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            suffix = Path(uploaded["filename"]).suffix or ".wav"
            upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
            upload_path.write_bytes(uploaded["content"])
            result["speech_path"] = str(upload_path)
            if mode in {"speech", "all"}:
                result["speech_prediction"] = recognizer.predict_speech(str(upload_path))
            if mode in {"fusion", "all"}:
                if not text:
                    raise ValueError("Text is required for fusion prediction.")
                result["fusion_prediction"] = recognizer.predict_fusion(str(upload_path), text)

        if mode not in {"speech", "text", "fusion", "all"}:
            raise ValueError("Mode must be speech, text, fusion, or all.")
        return result


def parse_multipart(body: bytes, boundary: bytes) -> tuple[dict[str, str], dict[str, dict]]:
    fields: dict[str, str] = {}
    files: dict[str, dict] = {}
    delimiter = b"--" + boundary
    for part in body.split(delimiter):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, _, content = part.partition(b"\r\n\r\n")
        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disposition = next(
            (header for header in headers if header.lower().startswith("content-disposition")),
            "",
        )
        params = parse_header_params(disposition)
        name = params.get("name")
        filename = params.get("filename")
        content = content.rstrip(b"\r\n")
        if not name:
            continue
        if filename:
            files[name] = {"filename": filename, "content": content}
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files


def parse_header_params(header: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for chunk in header.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        params[key.strip().lower()] = value.strip().strip('"')
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the emotion recognition API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), EmotionAPIHandler)
    print(f"Emotion API running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
