import logging
import re
from typing import List, Optional

import pdfplumber

logger = logging.getLogger(__name__)

TableData = List[List[Optional[str]]]


class TableExtractor:
    """Enhanced table extraction that preserves multi-line cell structure."""

    HEADER_HINTS = {
        "name",
        "title",
        "date",
        "description",
        "role",
        "company",
        "skill",
        "education",
        "experience",
        "year",
        "grade",
        "percentage",
        "course",
    }

    def extract_tables_preserve_structure(self, file_path: str) -> str:
        sections: List[str] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    try:
                        tables = page.extract_tables() or []
                    except Exception as ex:
                        logger.warning(
                            "Table extraction failed for page %s in %s: %s",
                            page_idx,
                            file_path,
                            ex,
                        )
                        continue

                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue
                        text = self._format_table_preserve_cells(table)
                        if text:
                            sections.append(f"[Page {page_idx + 1} Table {table_idx + 1}]\n{text}")
        except Exception as ex:
            logger.warning("Failed to extract tables from %s: %s", file_path, ex)
            return ""

        return "\n\n".join(sections).strip()

    def _format_table_preserve_cells(self, table: TableData) -> str:
        if not table:
            return ""

        has_header = self._is_header_row(table[0])
        headers = [self._clean_cell_text(str(c) if c is not None else "") for c in table[0]] if has_header else []
        out_lines: List[str] = []

        start_idx = 1 if has_header else 0
        if has_header:
            label_row = " | ".join(f"[{h}]" for h in headers if h)
            if label_row:
                out_lines.append(label_row)
                out_lines.append("-" * 50)

        for row in table[start_idx:]:
            if not row:
                continue
            if all(not self._clean_cell_text(str(c) if c is not None else "") for c in row):
                continue

            row_items: List[str] = []
            for col_idx, cell in enumerate(row):
                cell_text = self._clean_cell_text(str(cell) if cell is not None else "")
                if not cell_text:
                    continue

                if has_header and col_idx < len(headers) and headers[col_idx]:
                    value = self._format_multiline_cell(cell_text)
                    row_items.append(f"{headers[col_idx]}: {value}")
                else:
                    row_items.append(self._format_multiline_cell(cell_text))

            if row_items:
                out_lines.extend(f"  - {item}" for item in row_items)
                out_lines.append("")

        return "\n".join(out_lines).strip()

    def _is_header_row(self, first_row: List[Optional[str]]) -> bool:
        if not first_row:
            return False

        score = 0
        filled = 0
        for cell in first_row:
            text = self._clean_cell_text(str(cell) if cell is not None else "")
            if not text:
                continue
            filled += 1
            lower = text.lower()
            if any(h in lower for h in self.HEADER_HINTS):
                score += 2
            if len(lower.split()) <= 3:
                score += 1

        if filled == 0:
            return False
        return score >= filled

    def _format_multiline_cell(self, cell_text: str) -> str:
        lines = [ln.strip() for ln in cell_text.split("\n") if ln.strip()]
        if not lines:
            return ""
        if len(lines) == 1:
            return lines[0]

        formatted_lines: List[str] = []
        for line in lines:
            if line.startswith(("•", "-", "*", "✓", "→")):
                formatted_lines.append(line)
            else:
                formatted_lines.append(f"• {line}")
        return " ".join(formatted_lines)

    def _clean_cell_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
        return text.strip()


def format_table_to_text(table: TableData) -> str:
    """Backwards-compatible wrapper for existing callers."""
    return TableExtractor()._format_table_preserve_cells(table)


def extract_tables_from_pdf(file_path: str) -> str:
    return TableExtractor().extract_tables_preserve_structure(file_path)


def has_tables(file_path: str) -> bool:
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables() or []
                    if any(table for table in tables):
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False
