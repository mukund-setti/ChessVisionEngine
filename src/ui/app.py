"""FastAPI web application for Chess Vision Engine."""
# ruff: noqa: E501

from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chess_logic import BoardState, FENGenerator, PositionValidator
from src.detection import BoardDetector, PieceClassifier
from src.engine import StockfishWrapper
from src.utils.config import settings
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

engine: StockfishWrapper | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global engine
    setup_logging(settings.log_level)
    logger.info("Starting Chess Vision Engine")

    try:
        engine = StockfishWrapper()
        logger.info("Stockfish engine initialized")
    except FileNotFoundError:
        logger.warning("Stockfish not found - engine analysis disabled")
        engine = None

    yield

    if engine:
        engine.close()
        logger.info("Stockfish engine closed")


app = FastAPI(
    title="Chess Vision Engine",
    description="Scan chessboard photos and analyze positions",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


class AnalyzeRequest(BaseModel):
    fen: str
    depth: int = 20


class AnalyzeResponse(BaseModel):
    fen: str
    best_move: str
    score: str
    pv: list[str]
    is_valid: bool
    validation_errors: list[str]


class ScanResponse(BaseModel):
    fen: str
    confidence: float
    is_valid: bool
    validation_errors: list[str]
    board_ascii: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_available": engine is not None,
    }


@app.post("/api/scan", response_model=ScanResponse)
async def scan_board(request: Request):
    """Scan a chessboard image and return FEN."""
    contents = await _read_uploaded_image(request)
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        detector = BoardDetector()
        board = detector.detect_board_from_array(image)

        classifier = PieceClassifier()
        classification = classifier.classify_pieces(board)

        fen_gen = FENGenerator()
        fen_result = fen_gen.generate_with_validation(classification)

        board_state = BoardState.from_fen(fen_result.fen)

        return ScanResponse(
            fen=fen_result.fen,
            confidence=board.confidence,
            is_valid=fen_result.is_valid,
            validation_errors=fen_result.validation_errors,
            board_ascii=board_state.to_ascii(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


async def _read_uploaded_image(request: Request) -> bytes:
    """Read an uploaded image from multipart/form-data or a raw image body."""
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if not body:
        raise HTTPException(status_code=400, detail="Image file is required")

    if not content_type.startswith("multipart/form-data"):
        return body

    marker = "boundary="
    if marker not in content_type:
        raise HTTPException(status_code=400, detail="Multipart boundary is missing")

    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    boundary_bytes = f"--{boundary}".encode()

    for part in body.split(boundary_bytes):
        if b'name="file"' not in part:
            continue
        if b"\r\n\r\n" not in part:
            continue

        _, file_bytes = part.split(b"\r\n\r\n", 1)
        return file_bytes.rstrip(b"\r\n-")

    raise HTTPException(status_code=400, detail="Image file is required")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_position(request: AnalyzeRequest):
    """Analyze a chess position."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Chess engine not available",
        )

    validator = PositionValidator()
    validation = validator.validate(request.fen)

    if not validation.is_valid:
        return AnalyzeResponse(
            fen=request.fen,
            best_move="",
            score="",
            pv=[],
            is_valid=False,
            validation_errors=validation.errors,
        )

    try:
        result = engine.analyze(request.fen, depth=request.depth)
    except Exception as e:
        logger.exception("Engine analysis failed")
        raise HTTPException(
            status_code=500,
            detail=str(e) or "Engine analysis failed",
        ) from e

    return AnalyzeResponse(
        fen=request.fen,
        best_move=result.best_move,
        score=result.score,
        pv=result.pv,
        is_valid=True,
        validation_errors=[],
    )


@app.get("/api/validate")
async def validate_fen(fen: str):
    """Validate a FEN string."""
    validator = PositionValidator()
    result = validator.validate(fen)

    return {
        "fen": fen,
        "is_valid": result.is_valid,
        "is_legal": result.is_legal,
        "errors": result.errors,
        "warnings": result.warnings,
    }


@app.get("/api/legal-moves")
async def get_legal_moves(fen: str):
    """Get legal moves for a position."""
    try:
        board_state = BoardState.from_fen(fen)
        moves = board_state.legal_moves

        return {
            "fen": fen,
            "turn": board_state.turn,
            "is_check": board_state.is_check,
            "is_checkmate": board_state.is_checkmate,
            "moves": [
                {
                    "uci": m.uci,
                    "san": m.san,
                    "from": m.from_square,
                    "to": m.to_square,
                    "is_capture": m.is_capture,
                    "is_check": m.is_check,
                }
                for m in moves
            ],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main application page."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chess Vision Engine</title>
    <style>
        :root { --bg: #1a1a2e; --card: #16213e; --accent: #e94560; --text: #eaeaea; --muted: #8b8b9e; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; padding: 2rem; }
        h1 { font-size: 1.75rem; margin-bottom: 0.5rem; color: var(--accent); }
        .sub { color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }
        .card { background: var(--card); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
        .card h2 { font-size: 1rem; margin-bottom: 0.75rem; color: var(--muted); }
        label { display: block; margin-bottom: 0.5rem; font-weight: 500; }
        input[type="file"] { margin-bottom: 1rem; }
        button { background: var(--accent); color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 8px; cursor: pointer; font-weight: 600; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        button.secondary { background: #2d2d44; }
        pre, .fen-box { background: #0f0f1a; padding: 1rem; border-radius: 8px; font-family: monospace; white-space: pre-wrap; margin-top: 0.5rem; }
        .row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: flex-start; }
        .col { flex: 1; min-width: 200px; }
        .error { color: #e94560; margin-top: 0.5rem; }
        .success { color: #4ade80; margin-top: 0.5rem; }
        #boardAscii { font-size: 0.85rem; line-height: 1.2; }
        .api-link { color: var(--accent); text-decoration: none; font-size: 0.9rem; }
        .api-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chess Vision Engine</h1>
        <p class="sub">Upload a chessboard photo to scan the position, then analyze with Stockfish.</p>

        <div class="card">
            <h2>1. Scan board</h2>
            <label for="boardImage">Choose a chessboard image</label>
            <input type="file" id="boardImage" accept="image/*" />
            <button type="button" id="scanBtn" disabled>Scan</button>
            <div id="scanStatus"></div>
            <div id="scanResult" style="display:none;">
                <label>FEN</label>
                <div class="fen-box" id="scannedFen"></div>
                <label style="margin-top:1rem;">Board</label>
                <pre id="boardAscii"></pre>
            </div>
        </div>

        <div class="card">
            <h2>2. Analyze position</h2>
            <p style="margin-bottom:0.75rem;color:var(--muted);">Use the FEN from scan or paste any FEN.</p>
            <label for="fenInput">FEN</label>
            <input type="text" id="fenInput" style="width:100%;padding:0.6rem;background:#0f0f1a;border:1px solid #2d2d44;border-radius:8px;color:var(--text);margin-bottom:0.5rem;" placeholder="e.g. rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" />
            <button type="button" id="analyzeBtn" class="secondary">Analyze</button>
            <div id="analyzeStatus"></div>
            <div id="analyzeResult" style="display:none;">
                <div class="row">
                    <div class="col"><strong>Best move</strong><pre id="bestMove"></pre></div>
                    <div class="col"><strong>Score</strong><pre id="score"></pre></div>
                </div>
                <label style="margin-top:1rem;">Principal variation</label>
                <pre id="pv"></pre>
            </div>
        </div>

        <p style="margin-top:1.5rem;">
            <a class="api-link" href="/docs">OpenAPI docs</a> &middot;
            <a class="api-link" href="/health">Health</a>
        </p>
    </div>

    <script>
        const boardImage = document.getElementById('boardImage');
        const scanBtn = document.getElementById('scanBtn');
        const scanStatus = document.getElementById('scanStatus');
        const scanResult = document.getElementById('scanResult');
        const scannedFen = document.getElementById('scannedFen');
        const boardAscii = document.getElementById('boardAscii');
        const fenInput = document.getElementById('fenInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const analyzeStatus = document.getElementById('analyzeStatus');
        const analyzeResult = document.getElementById('analyzeResult');

        boardImage.addEventListener('change', () => { scanBtn.disabled = !boardImage.files.length; });

        scanBtn.addEventListener('click', async () => {
            if (!boardImage.files.length) return;
            scanBtn.disabled = true;
            scanStatus.textContent = 'Scanning...';
            scanStatus.className = '';
            const form = new FormData();
            form.append('file', boardImage.files[0]);
            try {
                const r = await fetch('/api/scan', { method: 'POST', body: form });
                const data = await r.json();
                if (!r.ok) throw new Error(data.detail || r.statusText);
                scannedFen.textContent = data.fen;
                boardAscii.textContent = data.board_ascii || '';
                fenInput.value = data.fen;
                scanResult.style.display = 'block';
                scanStatus.textContent = data.is_valid ? 'Position valid.' : 'Position has issues: ' + (data.validation_errors || []).join(', ');
                scanStatus.className = data.is_valid ? 'success' : 'error';
            } catch (e) {
                scanStatus.textContent = 'Error: ' + e.message;
                scanStatus.className = 'error';
            }
            scanBtn.disabled = false;
        });

        analyzeBtn.addEventListener('click', async () => {
            const fen = fenInput.value.trim();
            if (!fen) { analyzeStatus.textContent = 'Enter or scan a FEN first.'; analyzeStatus.className = 'error'; return; }
            analyzeBtn.disabled = true;
            analyzeStatus.textContent = 'Analyzing...';
            analyzeStatus.className = '';
            try {
                const r = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fen, depth: 20 })
                });
                const text = await r.text();
                let data = {};
                try {
                    data = text ? JSON.parse(text) : {};
                } catch (_) {
                    data = { detail: text || r.statusText };
                }
                if (!r.ok) throw new Error(data.detail || r.statusText || text);
                document.getElementById('bestMove').textContent = data.best_move || '-';
                document.getElementById('score').textContent = data.score || '-';
                document.getElementById('pv').textContent = (data.pv && data.pv.length) ? data.pv.join(' ') : '-';
                analyzeResult.style.display = 'block';
                analyzeStatus.textContent = 'Done.';
                analyzeStatus.className = 'success';
            } catch (e) {
                analyzeStatus.textContent = 'Error: ' + (e.message || String(e));
                analyzeStatus.className = 'error';
            }
            analyzeBtn.disabled = false;
        });
    </script>
</body>
</html>"""
    return HTMLResponse(html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ui.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
