from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {
            "status": "ok",
            "models": {
                "speech": (PROJECT_ROOT / "models" / "speech_pipeline" / "speech_model.npz").exists(),
                "text": (PROJECT_ROOT / "models" / "text_pipeline" / "text_model.json").exists(),
                "fusion": (PROJECT_ROOT / "models" / "fusion_pipeline" / "fusion_model.npz").exists(),
            },
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))
