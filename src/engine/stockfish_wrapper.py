"""Stockfish chess engine wrapper."""

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Result of engine analysis."""

    fen: str
    depth: int
    best_move: str
    ponder: str | None
    score: str
    score_cp: int | None
    score_mate: int | None
    pv: list[str]
    nodes: int
    time_ms: int

    @property
    def is_mate(self) -> bool:
        """Check if position has forced mate."""
        return self.score_mate is not None


class StockfishWrapper:
    """Wrapper for Stockfish chess engine."""

    def __init__(
        self,
        stockfish_path: str | Path | None = None,
        threads: int | None = None,
        hash_mb: int | None = None,
    ):
        self.stockfish_path = Path(stockfish_path or settings.stockfish_path)
        self.threads = threads or settings.engine_threads
        self.hash_mb = hash_mb or settings.engine_hash_mb
        self.engine: chess.engine.SimpleEngine | None = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Ensure engine is running."""
        if self.engine is None:
            self._start_engine()
        return self.engine

    def _start_engine(self) -> None:
        """Start Stockfish engine."""
        if not self.stockfish_path.exists():
            raise FileNotFoundError(
                f"Stockfish not found at {self.stockfish_path}. "
                "Run 'python scripts/download_stockfish.py' to download it."
            )

        logger.info(f"Starting Stockfish from {self.stockfish_path}")

        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.stockfish_path))
        self.engine.configure({
            "Threads": self.threads,
            "Hash": self.hash_mb,
        })

        logger.info(f"Stockfish started (threads={self.threads}, hash={self.hash_mb}MB)")

    def close(self) -> None:
        """Close the engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None
            logger.info("Stockfish closed")

    def __enter__(self) -> "StockfishWrapper":
        self._ensure_engine()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def analyze(
        self,
        fen: str,
        depth: int | None = None,
        time_limit: float | None = None,
        multipv: int = 1,
    ) -> AnalysisResult:
        """Analyze a position."""
        engine = self._ensure_engine()
        board = chess.Board(fen)

        depth = depth or settings.engine_depth

        if time_limit:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=depth)

        logger.info(f"Analyzing position (depth={depth}): {fen}")

        with engine.analysis(board, limit, multipv=multipv) as analysis:
            for info in analysis:
                pass

        info = analysis.info

        best_move = info.get("pv", [None])[0]
        pv = info.get("pv", [])
        score = info.get("score")

        if score:
            pov_score = score.relative
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                score_str = f"Mate in {abs(mate_in)}" if mate_in > 0 else f"Mated in {abs(mate_in)}"
                score_cp = None
                score_mate = mate_in
            else:
                cp = pov_score.score()
                score_str = f"{cp/100:+.2f}"
                score_cp = cp
                score_mate = None
        else:
            score_str = "0.00"
            score_cp = 0
            score_mate = None

        return AnalysisResult(
            fen=fen,
            depth=info.get("depth", depth),
            best_move=best_move.uci() if best_move else "",
            ponder=info.get("pv", [None, None])[1].uci() if len(info.get("pv", [])) > 1 else None,
            score=score_str,
            score_cp=score_cp,
            score_mate=score_mate,
            pv=[m.uci() for m in pv],
            nodes=info.get("nodes", 0),
            time_ms=int(info.get("time", 0) * 1000),
        )

    def get_best_move(
        self,
        fen: str,
        depth: int | None = None,
        time_limit: float | None = None,
    ) -> str:
        """Get best move for a position."""
        result = self.analyze(fen, depth=depth, time_limit=time_limit)
        return result.best_move

    def evaluate(self, fen: str, depth: int | None = None) -> float:
        """Get numerical evaluation of position."""
        result = self.analyze(fen, depth=depth)

        if result.is_mate:
            return 100.0 if result.score_mate > 0 else -100.0
        return result.score_cp / 100.0 if result.score_cp else 0.0

    def get_top_moves(
        self,
        fen: str,
        n: int = 3,
        depth: int | None = None,
    ) -> list[tuple[str, str]]:
        """Get top N moves for a position."""
        engine = self._ensure_engine()
        board = chess.Board(fen)
        depth = depth or settings.engine_depth

        limit = chess.engine.Limit(depth=depth)

        results = []
        with engine.analysis(board, limit, multipv=n) as analysis:
            for info in analysis:
                if "multipv" in info:
                    pv = info.get("pv", [])
                    score = info.get("score")

                    if pv and score:
                        move = pv[0].uci()
                        pov_score = score.relative

                        if pov_score.is_mate():
                            eval_str = f"M{pov_score.mate()}"
                        else:
                            eval_str = f"{pov_score.score()/100:+.2f}"

                        idx = info["multipv"] - 1
                        while len(results) <= idx:
                            results.append(None)
                        results[idx] = (move, eval_str)

        return [r for r in results if r is not None]