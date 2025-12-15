"""Tests for FEN generation."""

import pytest

from src.chess_logic.fen_generator import FENGenerator, FENResult
from src.chess_logic.position_validator import PositionValidator
from src.detection.piece_classifier import (
    BoardClassification,
    ClassificationResult,
    PieceType,
)


class TestFENGenerator:
    """Tests for FENGenerator class."""

    def create_empty_board_classification(self) -> BoardClassification:
        """Create a classification for an empty board."""
        squares = [
            ClassificationResult(
                piece=PieceType.EMPTY,
                confidence=1.0,
                square_index=i,
            )
            for i in range(64)
        ]
        return BoardClassification(squares=squares)

    def test_generate_empty_board(self):
        """Test FEN generation for empty board."""
        generator = FENGenerator()
        classification = self.create_empty_board_classification()

        fen = generator.generate(classification)

        assert fen.startswith("8/8/8/8/8/8/8/8")

    def test_piece_type_to_fen(self):
        """Test PieceType to FEN conversion."""
        assert PieceType.WHITE_KING.to_fen() == "K"
        assert PieceType.WHITE_QUEEN.to_fen() == "Q"
        assert PieceType.WHITE_ROOK.to_fen() == "R"
        assert PieceType.WHITE_BISHOP.to_fen() == "B"
        assert PieceType.WHITE_KNIGHT.to_fen() == "N"
        assert PieceType.WHITE_PAWN.to_fen() == "P"
        assert PieceType.BLACK_KING.to_fen() == "k"
        assert PieceType.BLACK_QUEEN.to_fen() == "q"
        assert PieceType.BLACK_ROOK.to_fen() == "r"
        assert PieceType.BLACK_BISHOP.to_fen() == "b"
        assert PieceType.BLACK_KNIGHT.to_fen() == "n"
        assert PieceType.BLACK_PAWN.to_fen() == "p"
        assert PieceType.EMPTY.to_fen() == ""


class TestPositionValidator:
    """Tests for PositionValidator class."""

    def test_validate_starting_position(self):
        """Test validation of starting position."""
        validator = PositionValidator()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        result = validator.validate(fen)

        assert result.is_valid
        assert result.is_legal
        assert len(result.errors) == 0

    def test_validate_invalid_fen(self):
        """Test validation of invalid FEN."""
        validator = PositionValidator()
        fen = "invalid-fen-string"

        result = validator.validate(fen)

        assert not result.is_valid
        assert len(result.errors) > 0