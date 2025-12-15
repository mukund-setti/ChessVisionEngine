"""Chess piece classification using neural networks."""

from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from src.utils.logging_config import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


class PieceType(Enum):
    """Chess piece types."""

    EMPTY = 0
    WHITE_KING = 1
    WHITE_QUEEN = 2
    WHITE_ROOK = 3
    WHITE_BISHOP = 4
    WHITE_KNIGHT = 5
    WHITE_PAWN = 6
    BLACK_KING = 7
    BLACK_QUEEN = 8
    BLACK_ROOK = 9
    BLACK_BISHOP = 10
    BLACK_KNIGHT = 11
    BLACK_PAWN = 12

    def to_fen(self) -> str:
        """Convert piece to FEN character."""
        mapping = {
            PieceType.EMPTY: "",
            PieceType.WHITE_KING: "K",
            PieceType.WHITE_QUEEN: "Q",
            PieceType.WHITE_ROOK: "R",
            PieceType.WHITE_BISHOP: "B",
            PieceType.WHITE_KNIGHT: "N",
            PieceType.WHITE_PAWN: "P",
            PieceType.BLACK_KING: "k",
            PieceType.BLACK_QUEEN: "q",
            PieceType.BLACK_ROOK: "r",
            PieceType.BLACK_BISHOP: "b",
            PieceType.BLACK_KNIGHT: "n",
            PieceType.BLACK_PAWN: "p",
        }
        return mapping[self]


@dataclass
class ClassificationResult:
    """Result of piece classification for a single square."""

    piece: PieceType
    confidence: float
    square_index: int

    @property
    def square_name(self) -> str:
        """Get algebraic notation for square (e.g., 'e4')."""
        row = 7 - (self.square_index // 8)
        col = self.square_index % 8
        return f"{chr(ord('a') + col)}{row + 1}"


@dataclass
class BoardClassification:
    """Complete classification of all 64 squares."""

    squares: list[ClassificationResult]

    def get_piece_at(self, square: str) -> ClassificationResult | None:
        """Get piece at given square (e.g., 'e4')."""
        for result in self.squares:
            if result.square_name == square:
                return result
        return None

    def to_array(self) -> NDArray[np.int8]:
        """Convert to 8x8 array of piece types."""
        board = np.zeros((8, 8), dtype=np.int8)
        for result in self.squares:
            row = result.square_index // 8
            col = result.square_index % 8
            board[row, col] = result.piece.value
        return board


class PieceClassifier:
    """Classifies chess pieces from square images."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float | None = None,
    ):
        self.model_path = Path(model_path or settings.model_path)
        self.confidence_threshold = confidence_threshold or settings.confidence_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the classification model."""
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}. Using placeholder classifier.")
            self.model = None
            return

        try:
            import onnxruntime as ort

            self.model = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using placeholder classifier.")
            self.model = None

    def classify_pieces(
        self,
        board_or_squares: "DetectedBoard | list[NDArray[np.uint8]]",
    ) -> BoardClassification:
        """Classify all pieces on the board."""
        from src.detection.board_detector import DetectedBoard

        if isinstance(board_or_squares, DetectedBoard):
            squares = board_or_squares.squares
        else:
            squares = board_or_squares

        if len(squares) != 64:
            raise ValueError(f"Expected 64 squares, got {len(squares)}")

        results = []
        for i, square_image in enumerate(squares):
            result = self._classify_square(square_image, i)
            results.append(result)

        return BoardClassification(squares=results)

    def _classify_square(
        self,
        square_image: NDArray[np.uint8],
        index: int,
    ) -> ClassificationResult:
        """Classify a single square."""
        if self.model is None:
            return self._placeholder_classify(square_image, index)

        input_tensor = self._preprocess(square_image)

        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})

        probabilities = outputs[0][0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return ClassificationResult(
            piece=PieceType(predicted_class),
            confidence=confidence,
            square_index=index,
        )

    def _preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess image for model input."""
        import cv2

        resized = cv2.resize(image, (64, 64))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))

        return np.expand_dims(transposed, axis=0)

    def _placeholder_classify(
        self,
        square_image: NDArray[np.uint8],
        index: int,
    ) -> ClassificationResult:
        """Placeholder classification based on color analysis."""
        gray = np.mean(square_image, axis=2)
        variance = np.var(gray)

        if variance > 1000:
            h, w = square_image.shape[:2]
            center = square_image[h//4:3*h//4, w//4:3*w//4]
            brightness = np.mean(center)

            if brightness > 127:
                piece = PieceType.WHITE_PAWN
            else:
                piece = PieceType.BLACK_PAWN
            confidence = 0.5
        else:
            piece = PieceType.EMPTY
            confidence = 0.8

        return ClassificationResult(
            piece=piece,
            confidence=confidence,
            square_index=index,
        )