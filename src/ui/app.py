"""FastAPI web application for Chess Vision Engine."""

import io
import base64
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2

from src.detection import BoardDetector, PieceClassifier
from src.chess_logic import FENGenerator, PositionValidator, BoardState
from src.engine import StockfishWrapper
from src.utils.config import settings
from src.utils.logging_config import setup_logging, get_logger

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
async def scan_board(file: UploadFile = File(...)):
    """Scan a chessboard image and return FEN."""
    contents = await file.read()
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
        raise HTTPException(status_code=400, detail=str(e))


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

    result = engine.analyze(request.fen, depth=request.depth)

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
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main application page."""
    html_content =