"""Chess position analysis utilities."""

from dataclasses import dataclass
from enum import Enum

import chess

from src.engine.stockfish_wrapper import StockfishWrapper, AnalysisResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PositionType(Enum):
    """Classification of position type."""

    WINNING = "winning"
    BETTER = "better"
    SLIGHT_EDGE = "slight_edge"
    EQUAL = "equal"
    UNCLEAR = "unclear"


@dataclass
class PositionAnalysis:
    """Comprehensive position analysis."""

    fen: str
    engine_result: AnalysisResult
    position_type: PositionType
    material_balance: int
    is_tactical: bool
    threats: list[str]
    weaknesses: list[str]
    suggested_plan: str


class PositionAnalyzer:
    """Analyzes chess positions comprehensively."""

    def __init__(self, engine: StockfishWrapper | None = None):
        self.engine = engine or StockfishWrapper()

    def analyze(self, fen: str, depth: int = 20) -> PositionAnalysis:
        """Perform comprehensive position analysis."""
        board = chess.Board(fen)

        engine_result = self.engine.analyze(fen, depth=depth)
        material = self._calculate_material(board)
        position_type = self._classify_position(engine_result)
        is_tactical = self._is_tactical(board, engine_result)
        threats = self._find_threats(board)
        weaknesses = self._find_weaknesses(board)
        plan = self._suggest_plan(board, engine_result, position_type)

        return PositionAnalysis(
            fen=fen,
            engine_result=engine_result,
            position_type=position_type,
            material_balance=material,
            is_tactical=is_tactical,
            threats=threats,
            weaknesses=weaknesses,
            suggested_plan=plan,
        )

    def _calculate_material(self, board: chess.Board) -> int:
        """Calculate material balance in centipawns."""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }

        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type in piece_values:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    material += value
                else:
                    material -= value

        return material

    def _classify_position(self, result: AnalysisResult) -> PositionType:
        """Classify position based on evaluation."""
        if result.is_mate:
            return PositionType.WINNING

        if result.score_cp is None:
            return PositionType.UNCLEAR

        cp = abs(result.score_cp)

        if cp >= 300:
            return PositionType.WINNING
        elif cp >= 150:
            return PositionType.BETTER
        elif cp >= 50:
            return PositionType.SLIGHT_EDGE
        else:
            return PositionType.EQUAL

    def _is_tactical(self, board: chess.Board, result: AnalysisResult) -> bool:
        """Check if position is tactical."""
        if result.is_mate:
            return True
        if board.is_check():
            return True
        if result.pv:
            first_move = chess.Move.from_uci(result.pv[0])
            if board.is_capture(first_move):
                return True
        return False

    def _find_threats(self, board: chess.Board) -> list[str]:
        """Find immediate threats in position."""
        threats = []

        for move in board.legal_moves:
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    attacker = board.piece_at(move.from_square)
                    piece_values = {
                        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
                    }
                    if piece_values.get(captured.piece_type, 0) > piece_values.get(attacker.piece_type, 0):
                        threats.append(
                            f"Win {chess.piece_name(captured.piece_type)} on {chess.square_name(move.to_square)}"
                        )

        for move in list(board.legal_moves)[:20]:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                threats.append(f"Checkmate: {move.uci()}")
                break
            board.pop()

        return threats[:5]

    def _find_weaknesses(self, board: chess.Board) -> list[str]:
        """Find positional weaknesses."""
        weaknesses = []

        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            pawns = board.pieces(chess.PAWN, color)

            for pawn_sq in pawns:
                fil