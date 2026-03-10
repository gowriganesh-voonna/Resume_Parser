import logging
import os
import time
from typing import Dict, Optional

import google.generativeai as genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiExtractor:
    """Extract text from images using Gemini vision-capable models."""

    MODELS = {
        "gemini-2.5-flash": {
            "rate_limit": 60,
            "context_length": 1_000_000,
            "supports_vision": True,
        },
        "gemini-1.5-flash": {
            "rate_limit": 60,
            "context_length": 1_000_000,
            "supports_vision": True,
        },
        "gemini-1.5-pro": {
            "rate_limit": 60,
            "context_length": 2_000_000,
            "supports_vision": True,
        },
    }

    def __init__(
        self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.last_request_time = 0.0
        self.min_request_interval = 1.0

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self._setup_model()

    def _setup_model(self):
        genai.configure(api_key=self.api_key)
        available_models = self._get_available_models()

        if available_models and not any(
            self.model_name in model.name for model in available_models
        ):
            logger.warning(
                "Model %s not found. Available: %s",
                self.model_name,
                [m.name for m in available_models],
            )
            for model in available_models:
                name = model.name.lower()
                if "flash" in name or "vision" in name:
                    self.model_name = model.name
                    logger.info("Falling back to model: %s", self.model_name)
                    break

        self.model = genai.GenerativeModel(self.model_name)
        logger.info("Initialized Gemini model: %s", self.model_name)

    def _get_available_models(self):
        try:
            models = genai.list_models()
            return [
                model
                for model in models
                if "generateContent"
                in getattr(model, "supported_generation_methods", [])
            ]
        except Exception as ex:
            logger.warning("Unable to list Gemini models: %s", ex)
            return []

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def extract_from_image(self, image_path: str, enhance_quality: bool = True) -> str:
        self._rate_limit()
        image = self._prepare_image(image_path)
        prompt = self._create_extraction_prompt(enhance_quality)

        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        response = self.model.generate_content(
            [prompt, image],
            generation_config=generation_config,
        )

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            logger.warning("Gemini returned empty response for %s", image_path)
            return self._fallback_extraction(image_path)

        cleaned = self._validate_and_clean(text)
        if not cleaned:
            logger.warning("Gemini response rejected by validation for %s", image_path)
            return self._fallback_extraction(image_path)

        logger.info("Gemini extracted %s chars from %s", len(cleaned), image_path)
        return cleaned

    def _rate_limit(self):
        now = time.time()
        delta = now - self.last_request_time
        if delta < self.min_request_interval:
            time.sleep(self.min_request_interval - delta)
        self.last_request_time = time.time()

    def _prepare_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _create_extraction_prompt(self, enhance_quality: bool) -> str:
        prompt = (
            "Extract all resume text exactly as shown. Preserve headings, bullets, dates, contact info, "
            "tables using | separators, and multi-column reading order (left-to-right then top-to-bottom). "
            "Return only extracted text."
        )
        if enhance_quality:
            prompt += " Pay extra attention to small/blurred text and keep structure faithful to source layout."
        return prompt

    def _validate_and_clean(self, text: str) -> str:
        lines = text.splitlines()
        cleaned_lines = []
        noisy = ("here is", "i have extracted", "the image shows", "based on the")
        for line in lines:
            s = line.strip()
            if s.startswith("```"):
                continue
            if len(s) < 100 and any(p in s.lower() for p in noisy):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        if len(cleaned) < 50:
            return ""
        return cleaned

    def _fallback_extraction(self, image_path: str) -> str:
        try:
            import pytesseract

            with Image.open(image_path) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")

                configs = ["--psm 1", "--psm 3", "--psm 6", "--psm 11"]
                texts = []
                for config in configs:
                    text = (
                        pytesseract.image_to_string(image, config=config, lang="eng")
                        or ""
                    ).strip()
                    if text:
                        texts.append(text)
                if texts:
                    return max(texts, key=len)
            return "[No text could be extracted from image]"
        except Exception as ex:
            logger.warning("Gemini fallback OCR failed: %s", ex)
            return "[Extraction failed - no methods available]"

    def extract_batch(
        self, image_paths: list[str], max_concurrent: int = 5
    ) -> Dict[str, str]:
        results: Dict[str, str] = {}
        for index, path in enumerate(image_paths):
            if index > 0 and index % max_concurrent == 0:
                time.sleep(2)
            try:
                results[path] = self.extract_from_image(path)
            except Exception as ex:
                logger.error("Batch extraction failed for %s: %s", path, ex)
                results[path] = f"[Error: {ex}]"
        return results
