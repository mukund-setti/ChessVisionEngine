"""Tests for board detection."""

import pytest
import numpy as np

from src.detection.board_detector import BoardDetector, DetectedBoard


class TestBoardDetector:
    """Tests for BoardDetector class."""

    def test_init_default_settings(self):
        """Test detector initializes with default settings."""
        detector = BoardDetector()
        assert detector.min_size == 200
        assert detector.max_size == 2000
        assert detector.method == "hough"

    def test_init_custom_settings(self):
        """Test detector with custom settings."""
        detector = BoardDetector(min_size=100, max_size=1000, method="contour")
        assert detector.min_size == 100
        assert detector.max_size == 1000
        assert detector.method == "contour"

    def test_detect_board_file_not_found(self):
        """Test error handling for missing file."""
        detector = BoardDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect_board("nonexistent.jpg")

    def test_order_corners(self):
        """Test corner ordering."""
        shuffled = np.array([
            [200, 200],
            [100, 100],
            [200, 100],
            [100, 200],
        ])

        ordered = BoardDetector._order_corners(shuffled)

        assert ordered[0][0] < ordered[1][0]
        assert ordered[0][1] < ordered[3][1]

    def test_extract_squares(self):
        """Test square extraction from board image."""
        detector = BoardDetector()
        board_image = np.zeros((800, 800, 3), dtype=np.uint8)

        squares = detector._extract_squares(board_image)

        assert len(squares) == 64
        assert all(sq.shape == (100, 100, 3) for sq in squares)


class TestDetectedBoard:
    """Tests for DetectedBoard dataclass."""

    def test_detected_board_creation(self):
        """Test DetectedBoard creation."""
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        corners = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
        squares = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(64)]

        board = DetectedBoard(
            image=image,
            corners=corners,
            squares=squares,
            confidence=0.95,
        )

        assert board.image.shape == (800, 800, 3)
        assert len(board.corners) == 4
        assert len(board.squares) == 64
        assert board.confidence == 0.95