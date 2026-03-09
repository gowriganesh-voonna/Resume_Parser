import os
import tempfile
import unittest
from unittest.mock import patch

try:
    from app.services.file_processor import FileProcessor
except ModuleNotFoundError as ex:
    raise unittest.SkipTest(f"Dependency missing for file processor tests: {ex}")


class TestFileProcessorPdf(unittest.TestCase):
    def test_layout_aware_fallback_to_legacy_pdf(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(b"fake")
            file_path = temp.name

        processor = FileProcessor()

        try:
            with patch(
                "app.services.file_processor.FileProcessor._extract_from_pdf",
                return_value="legacy text",
            ) as legacy_extract:
                with patch(
                    "app.services.layout_aware_extractor.LayoutAwareExtractor.extract_with_layout_awareness",
                    side_effect=RuntimeError("boom"),
                ):
                    text = processor.extract_text(file_path, "pdf")

            self.assertEqual(text, "legacy text")
            legacy_extract.assert_called_once_with(file_path)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


if __name__ == "__main__":
    unittest.main()
