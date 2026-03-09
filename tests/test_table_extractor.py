import unittest

try:
    from app.services.table_extractor import format_table_to_text
except ModuleNotFoundError as ex:
    raise unittest.SkipTest(f"Dependency missing for table extractor tests: {ex}")


class TestTableExtractor(unittest.TestCase):
    def test_format_table_with_headers(self):
        table = [
            ["Company", "Role"],
            ["Acme", "Engineer"],
            ["Globex", "Lead"],
        ]

        text = format_table_to_text(table)

        self.assertIn("Company: Acme", text)
        self.assertIn("Role: Engineer", text)
        self.assertIn("Company: Globex", text)

    def test_format_table_without_header(self):
        table = [
            ["Python", "Advanced"],
            ["SQL", "Intermediate"],
        ]

        text = format_table_to_text(table)

        self.assertIn("Python", text)
        self.assertIn("Advanced", text)

    def test_format_table_preserves_multiline_cell(self):
        table = [
            ["Company", "Responsibilities"],
            ["Acme", "Built APIs\nLed migration\nOn-call support"],
        ]

        text = format_table_to_text(table)

        self.assertIn("Responsibilities:", text)
        self.assertIn("Built APIs", text)
        self.assertIn("Led migration", text)


if __name__ == "__main__":
    unittest.main()
