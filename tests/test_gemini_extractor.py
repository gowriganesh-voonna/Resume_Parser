# test_gemini_extractor.py
import os
from app.services.ai_extractor import GeminiExtractor
import logging

logging.basicConfig(level=logging.INFO)


def test_extractor():
    # Initialize
    extractor = GeminiExtractor(model_name="gemini-2.5-flash")

    # Test with your screenshot
    test_image = r"C:\Users\voonn\testing_resumes\resume_sample.jpg"

    if os.path.exists(test_image):
        text = extractor.extract_from_image(test_image)
        print("\n" + "=" * 80)
        print("EXTRACTED TEXT:")
        print("=" * 80)
        print(text)
        print("=" * 80)
        print(f"\nLength: {len(text)} characters")
    else:
        print(f"Test image not found: {test_image}")


if __name__ == "__main__":
    test_extractor()
