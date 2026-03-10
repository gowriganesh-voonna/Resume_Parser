import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.services.image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    confidence: float = 0.0


class ImageLayoutAnalyzer:
    """Detect coarse layout signals for image OCR orchestration."""

    def __init__(self):
        self.min_column_width = 100
        self.column_gap_threshold = 50

    def analyze_layout(self, image: Image.Image) -> Dict:
        gray = self._to_gray(image)
        text_regions = self._detect_text_regions(gray)
        columns = self._detect_columns(text_regions, gray.shape[1])
        tables = self._detect_tables(gray)
        return {
            "width": gray.shape[1],
            "height": gray.shape[0],
            "column_count": len(columns),
            "columns": columns,
            "text_regions": text_regions,
            "has_tables": len(tables) > 0,
            "tables": tables,
        }

    def _to_gray(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _detect_text_regions(self, gray: np.ndarray) -> List[TextRegion]:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: List[TextRegion] = []
        h, w = gray.shape[:2]
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 20 or bh < 10:
                continue
            if bw > int(w * 0.9) and bh > int(h * 0.9):
                continue
            regions.append(TextRegion(x=x, y=y, width=bw, height=bh))

        regions.sort(key=lambda r: (r.y, r.x))
        return regions

    def _detect_columns(self, regions: List[TextRegion], image_width: int) -> List[Dict]:
        if not regions:
            return [{"x_start": 0, "x_end": image_width, "regions": []}]

        occupancy = np.zeros(image_width, dtype=np.float32)
        for region in regions:
            start = max(region.x, 0)
            end = min(region.x + region.width, image_width)
            occupancy[start:end] += 1.0

        if occupancy.max() > 0:
            occupancy = cv2.GaussianBlur(occupancy.reshape(1, -1), (1, 31), 0).flatten()

        active_threshold = max(0.5, float(occupancy.max()) * 0.08)
        gap_mask = occupancy <= active_threshold
        boundaries = [0]

        gap_start = None
        for idx, is_gap in enumerate(gap_mask):
            if is_gap and gap_start is None:
                gap_start = idx
            elif not is_gap and gap_start is not None:
                gap_width = idx - gap_start
                if gap_width >= self.column_gap_threshold:
                    boundaries.append(gap_start + gap_width // 2)
                gap_start = None

        if gap_start is not None:
            gap_width = image_width - gap_start
            if gap_width >= self.column_gap_threshold:
                boundaries.append(gap_start + gap_width // 2)

        boundaries.append(image_width)
        boundaries = sorted(set(boundaries))

        columns: List[Dict] = []
        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]
            if (x_end - x_start) < self.min_column_width:
                continue

            col_regions = [
                r
                for r in regions
                if (r.x + (r.width / 2.0)) >= x_start and (r.x + (r.width / 2.0)) < x_end
            ]
            if not col_regions:
                continue

            columns.append({"x_start": x_start, "x_end": x_end, "regions": col_regions})

        if not columns:
            return [{"x_start": 0, "x_end": image_width, "regions": regions}]

        columns.sort(key=lambda c: c["x_start"])
        return columns

    def _detect_tables(self, gray: np.ndarray) -> List[Dict]:
        edges = cv2.Canny(gray, 50, 150)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
        vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
        grid = cv2.bitwise_and(horizontal, vertical)

        if int(np.sum(grid > 0)) > 0:
            return [{"type": "grid_table", "confidence": 0.8}]
        return []

    def extract_by_column(self, image: Image.Image, column: Dict) -> Image.Image:
        rgb = image.convert("RGB")
        arr = np.array(rgb)
        width = arr.shape[1]
        x_start = max(0, min(int(column["x_start"]), width - 1))
        x_end = max(x_start + 1, min(int(column["x_end"]), width))
        cropped = arr[:, x_start:x_end]
        return Image.fromarray(cropped)


class ImageOCREnhancer:
    """Layout-aware OCR flow for images with robust fallback strategy."""

    def __init__(
        self,
        ocr_func: Optional[Callable[[Image.Image, Optional[str]], str]] = None,
    ):
        self.preprocessor = ImagePreprocessor()
        self.layout_analyzer = ImageLayoutAnalyzer()
        self.ocr_func = ocr_func

    def extract_text_with_layout(self, image: Image.Image) -> str:
        layout_info = self.layout_analyzer.analyze_layout(image)
        logger.info(
            "Image layout detected: columns=%s, has_tables=%s",
            layout_info["column_count"],
            layout_info["has_tables"],
        )

        if layout_info["column_count"] > 1:
            return self._extract_multi_column(image, layout_info).strip()
        if layout_info["has_tables"]:
            return self._extract_with_tables(image).strip()
        return self._extract_single_column(image).strip()

    def _extract_multi_column(self, image: Image.Image, layout_info: Dict) -> str:
        parts: List[str] = []
        for column in layout_info["columns"]:
            column_img = self.layout_analyzer.extract_by_column(image, column)
            candidate = self._extract_single_column(column_img)
            if candidate:
                parts.append(candidate)
        return "\n\n".join(parts)

    def _extract_with_tables(self, image: Image.Image) -> str:
        best_text = ""
        best_score = 0.0
        for candidate in self.preprocessor.enhance_with_multiple(image):
            text = self._ocr_with_table_awareness(candidate)
            score = self._score_extraction(text)
            if score > best_score:
                best_score = score
                best_text = text
        return best_text

    def _extract_single_column(self, image: Image.Image) -> str:
        best_text = ""
        best_score = -1
        for candidate in self.preprocessor.enhance_with_multiple(image):
            text = self._ocr_image(candidate)
            score = len((text or "").strip())
            if score > best_score:
                best_score = score
                best_text = text
        return (best_text or "").strip()

    def _ocr_image(self, image: Image.Image, config: Optional[str] = None) -> str:
        if self.ocr_func:
            try:
                return (self.ocr_func(image, config) or "").strip()
            except Exception as ex:
                logger.debug("Custom OCR callback failed: %s", ex)

        try:
            import pytesseract

            kwargs = {"lang": "eng"}
            if config:
                kwargs["config"] = config
            return (pytesseract.image_to_string(image, **kwargs) or "").strip()
        except Exception as ex:
            logger.debug("Tesseract OCR fallback failed: %s", ex)
            return ""

    def _ocr_with_table_awareness(self, image: Image.Image) -> str:
        configs = [
            "--psm 6",
            "--psm 11",
            "--psm 12",
            "-c preserve_interword_spaces=1",
        ]
        results: List[str] = []
        for cfg in configs:
            text = self._ocr_image(image, config=cfg)
            if text:
                results.append(text)
        if not results:
            return ""
        return max(results, key=len)

    def _score_extraction(self, text: str) -> float:
        if not text:
            return 0.0
        score = len(text) * 0.1
        if "\n\n" in text:
            score += 50.0
        if ": " in text:
            score += 30.0
        if text.count("\n") > 5:
            score += 20.0
        return score
