import logging
import os
import tempfile
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Tuple

import fitz
import pdfplumber
from PIL import Image

from app.config import settings
from app.services.table_extractor import TableExtractor, format_table_to_text

logger = logging.getLogger(__name__)

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class RegionType(Enum):
    TEXT_SINGLE_COL = "text_single_column"
    TEXT_MULTI_COL = "text_multi_column"
    TABLE = "table"
    IMAGE_SCREENSHOT = "image_screenshot"


class DocumentRegion:
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        region_type: RegionType,
        page_num: int,
        confidence: float,
    ):
        self.bbox = bbox
        self.region_type = region_type
        self.page_num = page_num
        self.confidence = confidence


class HybridLayoutExtractor:
    """Hybrid extraction for mixed PDF layouts with strong OCR fallback."""

    def __init__(self):
        self.table_extractor = TableExtractor()
        self.gemini_extractor = None
        if settings.GEMINI_API_KEY:
            try:
                from app.services.ai_extractor import GeminiExtractor

                self.gemini_extractor = GeminiExtractor(
                    api_key=settings.GEMINI_API_KEY,
                    model_name="gemini-2.5-flash",
                )
            except Exception as ex:
                logger.debug("Hybrid extractor Gemini init skipped: %s", ex)

    def extract_hybrid(self, file_path: str) -> Dict[str, Any]:
        # Fast path for scanned/screenshot-style PDFs.
        if self.is_screenshot_based_document(file_path):
            logger.info("Screenshot-based PDF detected. Using full-document OCR.")
            ocr_text = self.extract_screenshot_document(file_path)
            page_count = 0
            try:
                with fitz.open(file_path) as doc:
                    page_count = len(doc)
            except Exception:
                page_count = 0

            return {
                "full_text": ocr_text,
                "extraction_method": "ocr_full_document",
                "regions": [],
                "layout_map": {0: ["image_screenshot"]},
                "extraction_stats": {
                    "text_regions": 0,
                    "table_regions": 0,
                    "icon_regions": 0,
                    "image_regions": page_count,
                },
            }

        regions = self._segment_document(file_path)
        full_text_parts: List[str] = []
        out_regions: List[Dict[str, Any]] = []
        stats = {
            "text_regions": 0,
            "table_regions": 0,
            "icon_regions": 0,
            "image_regions": 0,
        }

        for region in regions:
            content = self._extract_region(file_path, region)
            out_regions.append(
                {
                    "type": region.region_type.value,
                    "bbox": region.bbox,
                    "page": region.page_num,
                    "content": content,
                    "confidence": region.confidence,
                }
            )

            if region.region_type == RegionType.TABLE:
                stats["table_regions"] += 1
            elif region.region_type == RegionType.IMAGE_SCREENSHOT:
                stats["image_regions"] += 1
            else:
                stats["text_regions"] += 1

            if content:
                marker = f"--- [{region.region_type.value}] Page {region.page_num + 1} ---"
                full_text_parts.append(marker)
                full_text_parts.append(content)

        return {
            "full_text": "\n".join(full_text_parts).strip(),
            "extraction_method": "hybrid_regions",
            "regions": out_regions,
            "layout_map": self._create_layout_map(regions),
            "extraction_stats": stats,
        }

    def is_screenshot_based_document(self, file_path: str) -> bool:
        """Detect pages with little text and image-heavy content."""
        try:
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
                if total_pages == 0:
                    return False

                pages_with_text = 0
                total_images = 0
                for page in doc:
                    text = (page.get_text("text") or "").strip()
                    if len(text) > 100:
                        pages_with_text += 1
                    total_images += len(page.get_images(full=True) or [])

            text_ratio = pages_with_text / total_pages
            return text_ratio < 0.3 and total_images > 0
        except Exception as ex:
            logger.error("Error detecting screenshot-based document: %s", ex)
            return False

    def extract_screenshot_document(self, file_path: str) -> str:
        pages_text: List[str] = []

        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    page_text = ""
                    for zoom in (3.0, 2.5, 2.0):
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat)
                        page_text = self._ocr_image(pix.tobytes("png")).strip()
                        if page_text:
                            break

                    if not page_text:
                        fallback = (page.get_text("text") or "").strip()
                        page_text = fallback if fallback else "[OCR failed]"

                    pages_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
        except Exception as ex:
            logger.error("Screenshot document OCR failed: %s", ex)
            return self._ocr_with_fallback(file_path)

        if not any(part.strip() and "[OCR failed]" not in part for part in pages_text):
            return self._ocr_with_fallback(file_path)

        return "\n\n".join(pages_text).strip()

    def _ocr_image(self, img_data: bytes) -> str:
        # Try Tesseract first for in-memory OCR.
        if TESSERACT_AVAILABLE:
            try:
                img = Image.open(BytesIO(img_data))
                gray = img.convert("L")
                text = pytesseract.image_to_string(gray, lang="eng")
                if text and text.strip():
                    return text.strip()
            except Exception as ex:
                logger.debug("Tesseract OCR failed for image bytes: %s", ex)
        return ""

    def _ocr_with_fallback(self, file_path: str) -> str:
        """Ultimate fallback OCR over all pages and zoom levels."""
        full_text: List[str] = []
        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    extracted = ""
                    for zoom in (2.0, 3.0, 4.0):
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                        extracted = self._ocr_image(pix.tobytes("png")).strip()
                        if extracted:
                            break
                    if not extracted:
                        extracted = "[OCR FAILED - Image could not be processed]"
                    full_text.append(f"--- Page {page_num + 1} ---\n{extracted}")
            return "\n\n".join(full_text).strip()
        except Exception as ex:
            logger.error("OCR fallback failed: %s", ex)
            return ""

    def _segment_document(self, file_path: str) -> List[DocumentRegion]:
        regions: List[DocumentRegion] = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                rect = page.rect
                regions.append(
                    DocumentRegion(
                        (rect.x0, rect.y0, rect.x1, rect.y1),
                        RegionType.TEXT_SINGLE_COL,
                        page_num,
                        0.6,
                    )
                )
                for bbox in self._detect_table_regions(file_path, page_num):
                    regions.append(DocumentRegion(bbox, RegionType.TABLE, page_num, 0.9))
                for bbox in self._detect_image_regions(page):
                    regions.append(DocumentRegion(bbox, RegionType.IMAGE_SCREENSHOT, page_num, 0.7))
        return regions

    def _detect_table_regions(self, file_path: str, page_num: int) -> List[Tuple[float, float, float, float]]:
        bboxes: List[Tuple[float, float, float, float]] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_num >= len(pdf.pages):
                    return bboxes
                page = pdf.pages[page_num]
                for table in page.find_tables().tables:
                    x0, y0, x1, y1 = table.bbox
                    bboxes.append((float(x0), float(y0), float(x1), float(y1)))
        except Exception as ex:
            logger.debug("Table region detection failed on page %s: %s", page_num, ex)
        return bboxes

    def _detect_image_regions(self, page: fitz.Page) -> List[Tuple[float, float, float, float]]:
        bboxes: List[Tuple[float, float, float, float]] = []
        for image in page.get_images(full=True):
            xref = image[0]
            try:
                for rect in page.get_image_rects(xref):
                    bboxes.append((rect.x0, rect.y0, rect.x1, rect.y1))
            except Exception:
                continue
        return bboxes

    def _extract_region(self, file_path: str, region: DocumentRegion) -> str:
        if region.region_type == RegionType.TABLE:
            return self._extract_table(file_path, region)
        if region.region_type == RegionType.IMAGE_SCREENSHOT:
            return self._extract_image_region(file_path, region)
        return self._extract_text_region(file_path, region)

    def _extract_table(self, file_path: str, region: DocumentRegion) -> str:
        try:
            with pdfplumber.open(file_path) as pdf:
                page = pdf.pages[region.page_num]
                tables = page.within_bbox(region.bbox).extract_tables() or []
                if tables and tables[0]:
                    return format_table_to_text(tables[0])
        except Exception as ex:
            logger.debug("Table extraction failed for region %s: %s", region.bbox, ex)

        # Fallback to whole-document table extraction if region crop fails.
        return self.table_extractor.extract_tables_preserve_structure(file_path)

    def _extract_image_region(self, file_path: str, region: DocumentRegion) -> str:
        enhanced = self._extract_screenshot_region(file_path, region)
        if enhanced:
            return enhanced

        # Preserve previous fallback behavior.
        try:
            with fitz.open(file_path) as doc:
                page = doc[region.page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=fitz.Rect(region.bbox))
            text = self._ocr_image(pix.tobytes("png"))
            return text if text else "[Screenshot region - OCR failed]"
        except Exception:
            return "[Screenshot region - OCR failed]"

    def _extract_screenshot_region(self, file_path: str, region: DocumentRegion) -> str:
        """Layout-aware OCR for screenshot-heavy regions with safe fallback."""
        try:
            with fitz.open(file_path) as doc:
                page = doc[region.page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=fitz.Rect(region.bbox))

            from app.services.image_layout_analyzer import ImageOCREnhancer

            with Image.open(BytesIO(pix.tobytes("png"))) as image:
                # Region OCR should stay local; avoid Gemini fan-out across many regions.
                enhancer = ImageOCREnhancer(ocr_func=self._ocr_pil_local)
                local_text = enhancer.extract_text_with_layout(image).strip()
                if self._is_region_text_sufficient(local_text):
                    return local_text

                # Single Gemini fallback for low-quality region OCR.
                gemini_text = self._ocr_region_with_gemini_once(image)
                if self._is_region_text_sufficient(gemini_text):
                    return gemini_text.strip()

                return local_text
        except Exception as ex:
            logger.debug("Enhanced screenshot region OCR failed: %s", ex)
            return ""

    def _ocr_pil_local(self, image: Image.Image, config: str | None = None) -> str:
        if TESSERACT_AVAILABLE:
            try:
                kwargs = {"lang": "eng"}
                if config:
                    kwargs["config"] = config
                return (pytesseract.image_to_string(image.convert("L"), **kwargs) or "").strip()
            except Exception as ex:
                logger.debug("Local tesseract OCR failed: %s", ex)
        return ""

    def _ocr_region_with_gemini_once(self, image: Image.Image) -> str:
        if not self.gemini_extractor:
            return ""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                image.convert("RGB").save(temp, format="PNG")
                temp_path = temp.name
            return (self.gemini_extractor.extract_from_image(temp_path) or "").strip()
        except Exception as ex:
            logger.debug("Gemini fallback for region OCR failed: %s", ex)
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def _is_region_text_sufficient(self, text: str, min_chars: int = 120) -> bool:
        cleaned = (text or "").strip()
        if len(cleaned) < min_chars:
            return False
        # Heuristic for heavily corrupted OCR outputs.
        alnum = sum(ch.isalnum() for ch in cleaned)
        return (alnum / max(len(cleaned), 1)) >= 0.55

    def _extract_text_region(self, file_path: str, region: DocumentRegion) -> str:
        with fitz.open(file_path) as doc:
            page = doc[region.page_num]
            return page.get_text("text", clip=fitz.Rect(region.bbox)).strip()

    def _create_layout_map(self, regions: List[DocumentRegion]) -> Dict[int, List[str]]:
        pages: Dict[int, set] = {}
        for region in regions:
            pages.setdefault(region.page_num, set()).add(region.region_type.value)
        return {page: sorted(list(types)) for page, types in pages.items()}
