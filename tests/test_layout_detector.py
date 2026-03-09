import unittest

try:
    from app.services.layout_detector import LayoutDetector, LayoutType
except ModuleNotFoundError as ex:
    raise unittest.SkipTest(f"Dependency missing for layout detector tests: {ex}")


class TestLayoutDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LayoutDetector()

    def test_classify_table_based(self):
        layout = self.detector._classify_layout(
            avg_columns=1.0,
            table_count=4,
            image_count=0,
            pages=[],
        )
        self.assertEqual(layout, LayoutType.TABLE_BASED)

    def test_classify_column_table_mix(self):
        layout = self.detector._classify_layout(
            avg_columns=2.0,
            table_count=1,
            image_count=0,
            pages=[],
        )
        self.assertEqual(layout, LayoutType.COLUMN_TABLE_MIX)

    def test_detect_columns_empty(self):
        self.assertEqual(self.detector._detect_columns([]), 1)


if __name__ == "__main__":
    unittest.main()
