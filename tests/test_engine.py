"""Tests for chess engine integration."""

import pytest
from unittest.mock import Mock

from src.engine.stockfish_wrapper import AnalysisResult
from src.engine.analysis import PositionAnalyzer, PositionType


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_is_mate_true(self):
        """Test is_mate property when mate exists."""
        result = AnalysisResult(
            fen="test",
            depth=20,
            best_move="e2e4",
            ponder=None,
            score="Mate in 5",
            score_cp=None,
            score_mate=5,
            pv=["e2e4"],
            nodes=1000,
            time_ms=100,
        )

        assert result.is_mate is True

    def test_is_mate_false(self):
        """Test is_mate property when no mate."""
        result = AnalysisResult(
            fen="test",
            depth=20,
            best_move="e2e4",
            ponder=None,
            score="+0.50",
            score_cp=50,
            score_mate=None,
            pv=["e2e4"],
            nodes=1000,
            time_ms=100,
        )

        assert result.is_mate is False


class TestPositionAnalyzer:
    """Tests for PositionAnalyzer class."""

    def test_classify_position_winning(self):
        """Test position classification for winning position."""
        analyzer = PositionAnalyzer.__new__(PositionAnalyzer)

        result = Mock()
        result.is_mate = False
        result.score_cp = 400

        position_type = analyzer._classify_position(result)

        assert position_type == PositionType.WINNING

    def test_classify_position_equal(self):
        """Test position classification for equal position."""
        analyzer = PositionAnalyzer.__new__(PositionAnalyzer)

        result = Mock()
        result.is_mate = False
        result.score_cp = 20

        position_type = analyzer._classify_position(result)

        assert position_type == PositionType.EQUAL

    def test_calculate_material_starting(self):
        """Test material calculation for starting position."""
        import chess

        analyzer = PositionAnalyzer.__new__(PositionAnalyzer)
        board = chess.Board()

        material = analyzer._calculate_material(board)

        assert material == 0