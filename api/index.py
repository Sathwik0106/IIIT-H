from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import EmotionRecognizer


recognizer = EmotionRecognizer(PROJECT_ROOT)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self.send_json(
                {
                    "status": "ok",
                    "models": {
                        "speech": (
                            PROJECT_ROOT / "models" / "speech_pipeline" / "speech_model.npz"
                        ).exists(),
                        "text": (
                            PROJECT_ROOT / "models" / "text_pipeline" / "text_model.json"
                        ).exists(),
                        "fusion": (
                            PROJECT_ROOT / "models" / "fusion_pipeline" / "fusion_model.npz"
                        ).exists(),
                    },
                }
            )
        return self.send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/predict/upload":
            return self.send_json({"error": "Not found"}, status=404)

        try:
            fields, files = parse_request(self)
            mode = fields.get("mode", "text")
            text = fields.get("text", "").strip()
            result = {"mode": mode, "text": text}

            if mode == "text":
                result["text_prediction"] = recognizer.predict_text(text)
            elif mode == "speech":
                audio_path = save_audio(files)
                result["speech_prediction"] = recognizer.predict_speech(str(audio_path))
            elif mode == "fusion":
                audio_path = save_audio(files)
                result["fusion_prediction"] = recognizer.predict_fusion(str(audio_path), text)
            else:
                raise ValueError("Mode must be speech, text, or fusion.")

            return self.send_json(result)
        except Exception as exc:
            return self.send_json({"error": str(exc)}, status=400)

    def send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


def parse_request(request: BaseHTTPRequestHandler):
    content_type = request.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data.")
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode("utf-8")
    body = request.rfile.read(int(request.headers.get("Content-Length", "0")))
    return parse_multipart(body, boundary)


def parse_multipart(body: bytes, boundary: bytes):
    fields = {}
    files = {}
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


def parse_header_params(header: str):
    params = {}
    for chunk in header.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        params[key.strip().lower()] = value.strip().strip('"')
    return params


def save_audio(files):
    uploaded = files.get("audio")
    if uploaded is None:
        raise ValueError("Audio file is required.")
    filename = uploaded["filename"].lower()
    if not filename.endswith((".wav", ".aiff", ".aif")):
        raise ValueError("Please upload a WAV or AIFF audio file.")
    suffix = Path(filename).suffix or ".wav"
    path = Path("/tmp") / f"{uuid4().hex}{suffix}"
    path.write_bytes(uploaded["content"])
    return path
