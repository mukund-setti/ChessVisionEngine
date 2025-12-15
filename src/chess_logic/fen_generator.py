"""FEN (Forsyth-Edwards Notation) generation from detected pieces."""

from dataclasses import dataclass

import chess

from src.detection.piece_classifier import BoardClassification, PieceType
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FENResult:
    """Result of FEN generation."""

    fen: str
    board: chess.Board
    is_valid: bool
    validation_errors: list[str]


class FENGenerator:
    """Generates FEN notation from board classification."""

    def __init__(self, assume_white_to_move: bool = True):
        self.assume_white_to_move = assume_white_to_move

    def generate(
        self,
        classification: BoardClassification,
        turn: str = "w",
        castling: str = "-",
        en_passant: str = "-",
        halfmove: int = 0,
        fullmove: int = 1,
    ) -> str:
        """Generate FEN string from board classification."""
        placement = self._build_placement(classification)
        fen = f"{placement} {turn} {castling} {en_passant} {halfmove} {fullmove}"

        logger.info(f"Generated FEN: {fen}")
        return fen

    def generate_with_validation(
        self,
        classification: BoardClassification,
        **kwargs,
    ) -> FENResult:
        """Generate FEN with validation."""
        fen = self.generate(classification, **kwargs)

        try:
            board = chess.Board(fen)
            is_valid = board.is_valid()
            errors = []

            if not is_valid:
                errors = self._get_validation_errors(board)

        except ValueError as e:
            board = chess.Board()
            is_valid = False
            errors = [str(e)]

        return FENResult(
            fen=fen,
            board=board,
            is_valid=is_valid,
            validation_errors=errors,
        )

    def _build_placement(self, classification: BoardClassification) -> str:
        """Build FEN piece placement string."""
        rows = []

        for rank in range(8):
            row_str = ""
            empty_count = 0

            for file in range(8):
                index = rank * 8 + file
                result = classification.squares[index]
                fen_char = result.piece.to_fen()

                if fen_char == "":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += fen_char

            if empty_count > 0:
                row_str += str(empty_count)

            rows.append(row_str)

        return "/".join(rows)

    def _get_validation_errors(self, board: chess.Board) -> list[str]:
        """Get list of validation errors for board position."""
        errors = []

        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))

        if white_kings != 1:
            errors.append(f"White has {white_kings} kings (should be 1)")
        if black_kings != 1:
            errors.append(f"Black has {black_kings} kings (should be 1)")

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                if rank == 0 or rank == 7:
                    errors.append(f"Pawn on invalid rank: {chess.square_name(square)}")

        if board.is_check():
            if not board.turn:
                errors.append("White is in check but it's black's turn")

        return errors

    def infer_castling_rights(self, classification: BoardClassification) -> str:
        """Infer castling rights from piece positions."""
        castling = ""

        white_king_pos = None
        for i, result in enumerate(classification.squares):
            if result.piece == PieceType.WHITE_KING:
                white_king_pos = i
                break

        if white_king_pos == 60:
            if classification.squares[63].piece == PieceType.WHITE_ROOK:
                castling += "K"
            if classification.squares[56].piece == PieceType.WHITE_ROOK:
                castling += "Q"

        black_king_pos = None
        for i, result in enumerate(classification.squares):
            if result.piece == PieceType.BLACK_KING:
                black_king_pos = i
                break

        if black_king_pos == 4:
            if classification.squares[7].piece == PieceType.BLACK_ROOK:
                castling += "k"
            if classification.squares[0].piece == PieceType.BLACK_ROOK:
                castling += "q"

        return castling if castling else "-"