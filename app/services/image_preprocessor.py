import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image enhancement helpers aimed at improving OCR quality."""

    def __init__(self):
        self.min_blur_variance = 100.0

    def enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Build multiple enhancement variants and select the best candidate
        using lightweight quality heuristics.
        """
        gray = self._to_gray(image)
        enhanced_versions: List[Tuple[str, np.ndarray]] = [
            ("gray", gray),
            ("contrast", self._enhance_contrast(gray)),
            ("denoised", self._denoise(gray)),
            ("sharpened", self._sharpen(gray)),
            ("adaptive", self._adaptive_threshold(gray)),
            ("morph", self._morphological_clean(gray)),
        ]

        if self._is_blurry(gray):
            enhanced_versions.append(("strong_sharpen", self._sharpen(gray, strength=2.0)))

        best = self._select_best_version(enhanced_versions)
        return Image.fromarray(cv2.cvtColor(best, cv2.COLOR_GRAY2RGB))

    def enhance_with_multiple(self, image: Image.Image) -> List[Image.Image]:
        """Return OCR-friendly variants for multi-pass extraction."""
        gray = self._to_gray(image)
        versions: List[np.ndarray] = [
            gray,
            self._enhance_contrast(gray),
            self._denoise(gray),
            self._sharpen(gray),
            self._adaptive_threshold(gray),
            self._morphological_clean(gray),
        ]
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions.append(binary)

        return [Image.fromarray(cv2.cvtColor(v, cv2.COLOR_GRAY2RGB)) for v in versions]

    def _to_gray(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    def _sharpen(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        kernel = np.array(
            [
                [-1, -1, -1],
                [-1, 9 * strength, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        ) / (strength + 7.0)
        return cv2.filter2D(img, -1, kernel)

    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def _morphological_clean(self, img: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _is_blurry(self, img: np.ndarray) -> bool:
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        return variance < self.min_blur_variance

    def _select_best_version(self, versions: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        best_name, best_img = versions[0]
        best_score = self._calculate_image_quality(best_img)

        for name, candidate in versions[1:]:
            score = self._calculate_image_quality(candidate)
            logger.debug("Image variant '%s' score: %.2f", name, score)
            if score > best_score:
                best_name, best_img, best_score = name, candidate, score

        logger.debug("Selected OCR image variant '%s' with score %.2f", best_name, best_score)
        return best_img

    def _calculate_image_quality(self, img: np.ndarray) -> float:
        contrast = float(img.std())
        edges = cv2.Canny(img, 50, 150)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)
        noise = float(np.var(cv2.medianBlur(img, 3).astype(np.float32) - img.astype(np.float32)))

        score = (contrast * 0.4) + (edge_density * 1000.0 * 0.4) - (noise * 0.2)
        return max(score, 0.0)
