"""Image preprocessing utilities."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Utilities for image preprocessing."""

    @staticmethod
    def load_image(path: str | Path) -> NDArray[np.uint8]:
        """Load image from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return image

    @staticmethod
    def resize(
        image: NDArray[np.uint8],
        max_size: int = 1024,
    ) -> NDArray[np.uint8]:
        """Resize image keeping aspect ratio."""
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image

        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def enhance_contrast(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Enhance image contrast using CLAHE."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Remove noise from image."""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    @staticmethod
    def to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def auto_orient(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Auto-orient board image so white is at bottom."""
        # TODO: Implement orientation detection
        return image