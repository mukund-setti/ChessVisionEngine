"""Board detection from images using computer vision."""

from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.logging_config import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


@dataclass
class DetectedBoard:
    """Represents a detected chessboard."""

    image: NDArray[np.uint8]
    corners: NDArray[np.float32]
    squares: list[NDArray[np.uint8]]
    confidence: float


class BoardDetector:
    """Detects and extracts chessboard from images."""

    def __init__(
        self,
        min_size: int | None = None,
        max_size: int | None = None,
        method: str | None = None,
    ):
        self.min_size = min_size or settings.min_board_size
        self.max_size = max_size or settings.max_board_size
        self.method = method or settings.board_detection_method

    def detect_board(self, image_path: str | Path) -> DetectedBoard:
        """Detect chessboard in an image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        logger.info(f"Processing image: {image_path.name} ({image.shape})")

        return self.detect_board_from_array(image)

    def detect_board_from_array(self, image: NDArray[np.uint8]) -> DetectedBoard:
        """Detect chessboard from numpy array."""
        if self.method == "hough":
            corners = self._detect_with_hough(image)
        elif self.method == "contour":
            corners = self._detect_with_contours(image)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

        if corners is None:
            raise ValueError("No chessboard detected in image")

        board_image = self._warp_perspective(image, corners)
        squares = self._extract_squares(board_image)

        return DetectedBoard(
            image=board_image,
            corners=corners,
            squares=squares,
            confidence=0.9,
        )

    def _detect_with_hough(self, image: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        """Detect board using Hough line detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        if lines is None:
            logger.warning("No lines detected")
            return None

        corners = self._find_corners_from_lines(lines, image.shape)
        return corners

    def _detect_with_contours(self, image: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        """Detect board using contour detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No contours detected")
            return None

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if self.min_size ** 2 <= area <= self.max_size ** 2:
                    return self._order_corners(approx.reshape(4, 2))

        return None

    def _find_corners_from_lines(
        self,
        lines: NDArray,
        shape: tuple[int, ...],
    ) -> NDArray[np.float32] | None:
        """Find board corners from detected lines."""
        horizontal = []
        vertical = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 30 or angle > 150:
                horizontal.append(line[0])
            elif 60 < angle < 120:
                vertical.append(line[0])

        if len(horizontal) < 2 or len(vertical) < 2:
            return None

        h_sorted = sorted(horizontal, key=lambda l: (l[1] + l[3]) / 2)
        v_sorted = sorted(vertical, key=lambda l: (l[0] + l[2]) / 2)

        top_line = h_sorted[0]
        bottom_line = h_sorted[-1]
        left_line = v_sorted[0]
        right_line = v_sorted[-1]

        corners = np.array([
            self._line_intersection(top_line, left_line),
            self._line_intersection(top_line, right_line),
            self._line_intersection(bottom_line, right_line),
            self._line_intersection(bottom_line, left_line),
        ], dtype=np.float32)

        return corners

    @staticmethod
    def _line_intersection(line1: NDArray, line2: NDArray) -> tuple[float, float]:
        """Calculate intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return ((x1 + x3) / 2, (y1 + y3) / 2)

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    @staticmethod
    def _order_corners(corners: NDArray) -> NDArray[np.float32]:
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]

        d = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(d)]
        rect[3] = corners[np.argmax(d)]

        return rect

    def _warp_perspective(
        self,
        image: NDArray[np.uint8],
        corners: NDArray[np.float32],
        size: int = 800,
    ) -> NDArray[np.uint8]:
        """Warp board to square top-down view."""
        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, matrix, (size, size))

        return warped

    def _extract_squares(
        self,
        board_image: NDArray[np.uint8],
    ) -> list[NDArray[np.uint8]]:
        """Extract 64 individual squares from board image."""
        h, w = board_image.shape[:2]
        square_h = h // 8
        square_w = w // 8

        squares = []
        for row in range(8):
            for col in range(8):
                y1 = row * square_h
                y2 = (row + 1) * square_h
                x1 = col * square_w
                x2 = (col + 1) * square_w

                square = board_image[y1:y2, x1:x2]
                squares.append(square)

        return squares