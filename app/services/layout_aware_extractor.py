import logging
from typing import Any, Dict, List

import fitz

from app.services.layout_detector import LayoutDetector, LayoutType
from app.services.table_extractor import extract_tables_from_pdf

logger = logging.getLogger(__name__)


class LayoutAwareExtractor:
    """Extract text with awareness of document layout."""

    def __init__(self):
        self.layout_detector = LayoutDetector()

    def extract_with_layout_awareness(self, file_path: str) -> Dict[str, Any]:
        layout_info = self.layout_detector.analyze_document(file_path)
        layout_type = layout_info.get("layout_type", LayoutType.SINGLE_COLUMN)

        result: Dict[str, Any] = {
            "layout_type": layout_type.value,
            "column_count": layout_info.get("column_count", 1),
            "has_tables": layout_info.get("has_tables", False),
            "has_images": layout_info.get("has_images", False),
            "text": "",
            "tables_text": "",
            "extraction_method": "standard",
        }

        try:
            if layout_type == LayoutType.TABLE_BASED:
                result["tables_text"] = self._extract_tables(file_path)
                result["text"] = self._extract_regular_text(file_path)
                result["extraction_method"] = "table_first"
            elif layout_type in [LayoutType.TWO_COLUMN, LayoutType.MULTI_COLUMN]:
                result["text"] = self._extract_multi_column(file_path, layout_info)
                result["extraction_method"] = "column_aware"
            elif layout_type == LayoutType.COLUMN_TABLE_MIX:
                result.update(self._extract_column_table_mix(file_path, layout_info))
                result["extraction_method"] = "column_table_mix"
            elif layout_type == LayoutType.IMAGE_GRAPHIC:
                result["text"] = self._extract_with_ocr(file_path)
                result["extraction_method"] = "ocr_based"
            elif layout_type == LayoutType.NESTED_COLUMNS:
                result["text"] = self._extract_nested_columns(file_path)
                result["extraction_method"] = "nested_columns"
            else:
                result["text"] = self._extract_standard(file_path)
                result["extraction_method"] = "standard"

            if result["has_tables"] and not result["tables_text"]:
                result["tables_text"] = self._extract_tables(file_path)

            return result
        except Exception as ex:
            logger.warning("Layout-aware extraction failed for %s: %s", file_path, ex)
            return {
                **result,
                "text": self._extract_standard(file_path),
                "tables_text": result.get("tables_text", ""),
                "extraction_method": "fallback_standard",
            }

    def _extract_multi_column(self, file_path: str, layout_info: Dict[str, Any]) -> str:
        full_text: List[str] = []

        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                page_layout = layout_info.get("pages", [])
                col_count = 1
                if page_num < len(page_layout):
                    col_count = max(1, int(page_layout[page_num].get("column_count", 1)))

                if col_count <= 1:
                    text = page.get_text("text").strip()
                    if text:
                        full_text.append(text)
                    continue

                rect = page.rect
                col_width = rect.width / col_count
                col_texts: List[str] = []
                for col_idx in range(col_count):
                    col_rect = fitz.Rect(
                        col_idx * col_width,
                        0,
                        (col_idx + 1) * col_width,
                        rect.height,
                    )
                    col_text = page.get_text("text", clip=col_rect).strip()
                    if col_text:
                        col_texts.append(col_text)

                if col_texts:
                    full_text.append("\n\n".join(col_texts))

        return "\n\n".join(full_text).strip()

    def _extract_column_table_mix(self, file_path: str, layout_info: Dict[str, Any]) -> Dict[str, str]:
        try:
            from app.services.hybrid_extractor import HybridLayoutExtractor

            hybrid = HybridLayoutExtractor()
            hybrid_result = hybrid.extract_hybrid(file_path)
            if hybrid_result.get("full_text"):
                return {
                    "text": hybrid_result["full_text"],
                    "tables_text": self._extract_tables(file_path),
                }
        except Exception as ex:
            logger.debug("Hybrid extractor unavailable for mixed layout: %s", ex)

        return {
            "text": self._extract_multi_column(file_path, layout_info),
            "tables_text": self._extract_tables(file_path),
        }

    def _extract_nested_columns(self, file_path: str) -> str:
        pages: List[str] = []

        with fitz.open(file_path) as doc:
            for page in doc:
                blocks = page.get_text("dict").get("blocks", [])
                sorted_blocks = sorted(
                    [b for b in blocks if "lines" in b and "bbox" in b],
                    key=lambda b: (b["bbox"][1], b["bbox"][0]),
                )

                block_texts: List[str] = []
                for block in sorted_blocks:
                    lines: List[str] = []
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(span.get("text", "") for span in spans).strip()
                        if line_text:
                            lines.append(line_text)
                    if lines:
                        block_texts.append(" ".join(lines))

                if block_texts:
                    pages.append("\n".join(block_texts))

        return "\n\n".join(pages).strip()

    def _extract_tables(self, file_path: str) -> str:
        return extract_tables_from_pdf(file_path)

    def _extract_standard(self, file_path: str) -> str:
        with fitz.open(file_path) as doc:
            texts = [page.get_text("text") for page in doc]
        return "\n".join(t for t in texts if t).strip()

    def _extract_with_ocr(self, file_path: str) -> str:
        from app.services.file_processor import FileProcessor

        processor = FileProcessor()
        return processor._extract_from_image(file_path)

    def _extract_regular_text(self, file_path: str) -> str:
        return self._extract_standard(file_path)
