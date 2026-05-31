"""Run a local end-to-end smoke check for the web API."""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
import uuid
from io import BytesIO

import uvicorn
from PIL import Image, ImageDraw

from src.ui.app import app


def _read_json(url: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _make_board_image() -> bytes:
    image = Image.new("RGB", (256, 256), "#f0d9b5")
    draw = ImageDraw.Draw(image)
    colors = ("#f0d9b5", "#b58863")

    for row in range(8):
        for col in range(8):
            x1 = col * 32
            y1 = row * 32
            x2 = x1 + 32
            y2 = y1 + 32
            draw.rectangle([x1, y1, x2, y2], fill=colors[(row + col) % 2])

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _post_multipart_image(url: str, image_bytes: bytes, timeout: int = 30) -> dict:
    boundary = f"----codex{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="board.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    body += image_bytes
    body += f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    port = 8097
    base_url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        for _ in range(100):
            try:
                health = _read_json(f"{base_url}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("Server did not start")

        if health["status"] != "healthy" or not health["engine_available"]:
            raise AssertionError(f"Unexpected health response: {health}")
        print(f"health ok: {health}")

        with urllib.request.urlopen(f"{base_url}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        if "Chess Vision Engine" not in html:
            raise AssertionError("UI did not render the expected title")
        print(f"ui ok: {len(html)} bytes")

        scan = _post_multipart_image(
            f"{base_url}/api/scan",
            _make_board_image(),
            timeout=30,
        )
        if not scan["is_valid"] or not scan["fen"]:
            raise AssertionError(f"Unexpected scan response: {scan}")
        print(f"scan ok: {scan['fen']}")

        analysis = _post_json(
            f"{base_url}/api/analyze",
            {"fen": scan["fen"], "depth": 1},
            timeout=30,
        )
        if not analysis["is_valid"] or not analysis["best_move"]:
            raise AssertionError(f"Unexpected analysis response: {analysis}")
        print(f"analysis ok: {analysis['best_move']} {analysis['score']}")

        fen = urllib.parse.quote(scan["fen"])
        validation = _read_json(f"{base_url}/api/validate?fen={fen}", timeout=5)
        if not validation["is_valid"]:
            raise AssertionError(f"Unexpected validation response: {validation}")
        print("validation ok")

        legal_moves = _read_json(f"{base_url}/api/legal-moves?fen={fen}", timeout=5)
        if not legal_moves["moves"]:
            raise AssertionError(f"Unexpected legal moves response: {legal_moves}")
        print(f"legal moves ok: {len(legal_moves['moves'])} moves")

    finally:
        server.should_exit = True
        thread.join(timeout=10)
        print("server stopped")


if __name__ == "__main__":
    main()
