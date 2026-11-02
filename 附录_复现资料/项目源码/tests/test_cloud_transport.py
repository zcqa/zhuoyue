"""Local HTTP/WebSocket contract tests. These never call a paid provider."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import numpy as np
import soundfile as sf
from websockets.sync.server import serve
from meeting_asr import cloud


def fake_result(text="我们测试", final=True):
    return {"type": "Results", "start": 0.0, "duration": .2, "is_final": final, "speech_final": final,
            "channel": {"alternatives": [{"transcript": text, "words": [
                {"word": text, "start": 0.0, "end": .2, "speaker": 0}]}]}}


def test_batch_actual_http_transport_to_local_fixture(monkeypatch, tmp_path):
    payload = {"metadata": {"duration": .2}, "results": {"channels": [fake_result()["channel"]]}}
    observed = {}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed["path"] = self.path
            observed["auth"] = self.headers.get("Authorization")
            observed["audio"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros(3200, np.float32), 16000)
    monkeypatch.setattr(cloud, "BATCH_URL", f"http://127.0.0.1:{server.server_port}/listen")
    monkeypatch.setattr(cloud, "key_or_error", lambda: "test-only-placeholder")
    try:
        result = cloud.run_batch(path, tmp_path / "output")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    assert "diarize_model=latest" in observed["path"]
    assert observed["auth"] == "Token test-only-placeholder"
    assert observed["audio"][:4] == b"RIFF"
    assert result["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert "test-only-placeholder" not in (tmp_path / "output/request_plan.json").read_text()


def test_stream_actual_websocket_transport_to_local_fixture(monkeypatch, tmp_path):
    received = []
    def handler(socket):
        for message in socket:
            received.append(message)
            if isinstance(message, bytes):
                socket.send(json.dumps(fake_result("我们", final=False)))
            elif json.loads(message).get("type") == "CloseStream":
                socket.send(json.dumps(fake_result("我们测试", final=True)))
                socket.close()
                break
    server = serve(handler, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(cloud, "STREAM_URL", f"ws://127.0.0.1:{port}/listen")
    monkeypatch.setattr(cloud, "key_or_error", lambda: "test-only-placeholder")
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros(3200, np.float32), 16000)
    try:
        result = cloud.run_cloud_stream(path, tmp_path / "output", duration=0)
    finally:
        server.shutdown()
        thread.join(2)
    events = [json.loads(line) for line in (tmp_path / "output/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(isinstance(m, bytes) for m in received)
    assert result["status"] == "completed"
    assert len({e["segment_id"] for e in events}) == 1
    assert events[-1]["state"] == "final" and events[-1]["text"] == "我们测试"
