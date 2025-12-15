"""Chess position validation."""

from dataclasses import dataclass

import chess

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of position validation."""

    is_valid: bool
    is_legal: bool
    errors: list[str]
    warnings: list[str]


class PositionValidator:
    """Validates chess positions."""

    def validate(self, fen: str) -> ValidationResult:
        """Validate a FEN position."""
        errors = []
        warnings = []

        try:
            board = chess.Board(fen)
        except ValueError as e:
            return ValidationResult(
                is_valid=False,
                is_legal=False,
                errors=[f"Invalid FEN: {e}"],
                warnings=[],
            )

        # Check king counts
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))

        if white_kings != 1:
            errors.append(f"Invalid white king count: {white_kings}")
        if black_kings != 1:
            errors.append(f"Invalid black king count: {black_kings}")

        # Check pawn positions
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            for square in pawns:
                rank = chess.square_rank(square)
                if rank == 0 or rank == 7:
                    errors.append(f"Pawn on invalid rank: {chess.square_name(square)}")

        # Check pawn counts
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))

        if white_pawns > 8:
            warnings.append(f"White has {white_pawns} pawns (max 8)")
        if black_pawns > 8:
            warnings.append(f"Black has {black_pawns} pawns (max 8)")

        # Check total piece counts
        for color, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            total = (
                len(board.pieces(chess.PAWN, color)) +
                len(board.pieces(chess.KNIGHT, color)) +
                len(board.pieces(chess.BISHOP, color)) +
                len(board.pieces(chess.ROOK, color)) +
                len(board.pieces(chess.QUEEN, color)) +
                len(board.pieces(chess.KING, color))
            )
            if total > 16:
                warnings.append(f"{name} has {total} pieces (max 16)")

        # Check if side not to move is in check
        board_copy = board.copy()
        board_copy.turn = not board.turn
        if board_copy.is_check():
            errors.append("Side not to move is in check (impossible position)")

        is_valid = len(errors) == 0
        is_legal = is_valid and board.is_valid()

        return ValidationResult(
            is_valid=is_valid,
            is_legal=is_legal,
            errors=errors,
            warnings=warnings,
        )

    def suggest_corrections(self, fen: str) -> list[str]:
        """Suggest corrections for an invalid position."""
        suggestions = []
        validation = self.validate(fen)

        if validation.is_valid:
            return []

        for error in validation.errors:
            if "king count" in error.lower():
                suggestions.append(
                    "Check if a king was misidentified as another piece or vice versa"
                )
            if "pawn on invalid rank" in error.lower():
                suggestions.append(
                    "Pawns on ranks 1 or 8 should be promoted pieces"
                )
            if "side not to move is in check" in error.lower():
                suggestions.append(
                    "Try switching which side is to move"
                )

        return suggestion