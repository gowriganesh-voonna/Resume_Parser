import logging
from enum import Enum
from typing import Any, Dict, List

import fitz
import numpy as np
import pdfplumber

from app.services.table_extractor import extract_tables_from_pdf

logger = logging.getLogger(__name__)


class LayoutType(Enum):
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    MULTI_COLUMN = "multi_column"
    TABLE_BASED = "table_based"
    COLUMN_TABLE_MIX = "column_table_mix"
    ICON_BASED = "icon_based"
    IMAGE_GRAPHIC = "image_graphic"
    NESTED_COLUMNS = "nested_columns"


class LayoutDetector:
    """Detect document layout to choose extraction strategy."""

    def __init__(self):
        self.table_detector = TableDetector()

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        layout_info: Dict[str, Any] = {
            "layout_type": LayoutType.SINGLE_COLUMN,
            "has_tables": False,
            "has_images": False,
            "column_count": 1,
            "pages": [],
        }

        total_columns: List[int] = []
        total_tables = 0
        total_images = 0

        try:
            with fitz.open(file_path) as doc:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(doc):
                        page_layout = self._analyze_page(page, page_num, pdf)
                        total_columns.append(page_layout["column_count"])
                        total_tables += page_layout["table_count"]
                        total_images += page_layout["image_count"]
                        layout_info["pages"].append(page_layout)
        except Exception as ex:
            logger.warning("Layout analysis failed for %s: %s", file_path, ex)
            return layout_info

        avg_columns = float(np.mean(total_columns)) if total_columns else 1.0
        layout_info["column_count"] = max(1, int(round(avg_columns)))
        layout_info["has_tables"] = total_tables > 0
        layout_info["has_images"] = total_images > 0
        layout_info["layout_type"] = self._classify_layout(
            avg_columns,
            total_tables,
            total_images,
            layout_info["pages"],
        )

        return layout_info

    def _analyze_page(self, page: fitz.Page, page_num: int, pdf) -> Dict[str, Any]:
        blocks = page.get_text("dict").get("blocks", [])

        x_positions: List[float] = []
        for block in blocks:
            if "lines" in block and "bbox" in block:
                x_positions.append(float(block["bbox"][0]))

        column_count = self._detect_columns(x_positions)

        table_count = 0
        if page_num < len(pdf.pages):
            try:
                page_tables = pdf.pages[page_num].find_tables()
                table_count = len(page_tables.tables)
            except Exception:
                try:
                    extracted = pdf.pages[page_num].extract_tables() or []
                    table_count = len([t for t in extracted if t])
                except Exception:
                    table_count = 0

        image_count = len(page.get_images(full=True) or [])

        return {
            "page_num": page_num,
            "column_count": column_count,
            "table_count": table_count,
            "image_count": image_count,
            "has_nested_columns": self._detect_nested_columns(blocks),
        }

    def _detect_columns(self, x_positions: List[float]) -> int:
        if not x_positions:
            return 1

        unique = sorted(set(x_positions))
        if len(unique) < 2:
            return 1

        gaps = np.diff(unique)
        boundaries = np.where(gaps > 100)[0]
        return max(1, int(len(boundaries) + 1))

    def _detect_nested_columns(self, blocks: List[Dict[str, Any]]) -> bool:
        nested_count = 0

        for block in blocks:
            lines = block.get("lines")
            if not lines or len(lines) < 4:
                continue

            indentations = []
            for line in lines:
                spans = line.get("spans", [])
                if spans:
                    origin = spans[0].get("origin", [0.0, 0.0])
                    indentations.append(float(origin[0]))

            if len(set(indentations)) > 1:
                nested_count += 1

        return nested_count > 2

    def _classify_layout(
        self,
        avg_columns: float,
        table_count: int,
        image_count: int,
        pages: List[Dict[str, Any]],
    ) -> LayoutType:
        if table_count > 3 and avg_columns <= 1.5:
            return LayoutType.TABLE_BASED

        if image_count > 5 and table_count == 0:
            return LayoutType.IMAGE_GRAPHIC

        if table_count > 0 and avg_columns > 1:
            return LayoutType.COLUMN_TABLE_MIX

        if any(p.get("has_nested_columns", False) for p in pages):
            return LayoutType.NESTED_COLUMNS

        if avg_columns >= 3:
            return LayoutType.MULTI_COLUMN
        if avg_columns >= 2:
            return LayoutType.TWO_COLUMN

        return LayoutType.SINGLE_COLUMN


class TableDetector:
    """Dedicated table extraction wrapper."""

    def extract_tables(self, file_path: str) -> str:
        return extract_tables_from_pdf(file_path)
