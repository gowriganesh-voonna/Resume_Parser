from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

import google.generativeai as genai
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models.poc1_models import (
    Domain,
    ParseStatus,
    POC1Output,
    PersonalInfo,
    StorageInfo,
)

logger = logging.getLogger(__name__)


class AIResumeProcessor:
    """Convert raw extracted text into POC1 structured output via Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.valid_domains = [d.value for d in Domain]

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def process_resume(
        self,
        raw_text: str,
        candidate_id: str,
        file_name: str = "",
    ) -> POC1Output:
        logger.info("Processing resume with AI: %s", candidate_id)
        prompt = self._build_processing_prompt(raw_text, candidate_id, file_name)

        try:
            response = self.model.generate_content(prompt)
            response_text = (getattr(response, "text", "") or "").strip()
            if not response_text:
                raise ValueError("Empty response from AI")

            parsed = self._parse_ai_response(response_text)
            result = self._validate_and_enhance(parsed, candidate_id, raw_text=raw_text)
            logger.info(
                "AI profile generated for %s with confidence %.2f",
                candidate_id,
                result.confidence_score,
            )
            return result
        except Exception as ex:
            logger.error("AI processing failed for %s: %s", candidate_id, ex)
            return self._create_fallback_output(raw_text, candidate_id, str(ex))

    #     def _build_processing_prompt(self, raw_text: str, candidate_id: str, file_name: str) -> str:
    #         safe_text = raw_text[:120000]
    #         return f"""
    # You are an expert resume parser for SmartRecruitz POC1.
    # Input may originate from PDF, DOCX, TXT, scanned image OCR, or mixed layout extraction.
    # Return only JSON matching the schema exactly.

    # Candidate ID: {candidate_id}
    # File Name: {file_name}

    # Allowed domain values: {", ".join(self.valid_domains)}
    # Allowed proficiency values: Beginner, Intermediate, Advanced, Expert

    # Top-level keys required exactly:
    # candidate_id, parse_status, confidence_score, personal_info, experience, education,
    # skills_raw, skills_normalized, implied_skills, total_experience_years, primary_domain,
    # domain_wise_experience, certifications, projects, languages, extra_data,
    # parsing_warnings, duplicate_pre_check, storage, raw_text

    # Critical output type rules:
    # 1) Return JSON only, no markdown, no prose.
    # 2) skills_raw must be array of strings only. Never a single string.
    # 3) skills_normalized must be array of objects only:
    #    {{standard_name, original_terms[], proficiency, evidence, years_experience}}
    # 4) implied_skills must be array of objects only:
    #    {{skill, inferred_from, confidence}}
    # 5) projects must be array of objects and MUST use key "name" (never "title"):
    #    {{name, description, technologies[], domain, url, duration}}
    # 6) duplicate_pre_check must always be an object, never null.
    # 7) If missing data, use null or empty arrays, but preserve valid types.
    # 8) extra_data must always be an object with keys:
    #    awards[], publications[], links[], others[]

    # Official required-field rules:
    # 1) personal_info.full_name is required when reasonably inferable.
    # 2) At least one of: experience[] not empty OR education[] not empty.
    # 3) If both are unavailable, set parse_status="FAILED" and add clear parsing_warnings.

    # Experience extraction requirements (strict):
    # 1) experience[] objects must include at least company and title.
    # 2) Use keys exactly: company, title, start_date, end_date, duration_months, domain, responsibilities, is_current, location
    # 3) Normalize end_date to "Present" for current job.
    # 4) If dates are present, compute duration_months (positive integer).
    # 5) If fresher/no jobs, set experience=[] and total_experience_years=0.
    # 6) Prefer chronology and avoid overlap errors; add warnings in parsing_warnings if uncertain.
    # 7) domain per experience must be one allowed domain, else "Other".
    # 8) Validate no impossible timelines (future end dates, negative durations). If uncertain, keep dates but add warning.
    # 9) Compute total_experience_years from validated durations.
    # 10) Compute domain_wise_experience with fields:
    #     domain, experience_years, experience_months, companies[], roles[]
    # 11) Ensure sum(domain_wise_experience.experience_months) ~= total experience months; if mismatch, fix and warn.
    # 12) primary_domain must be domain with highest validated months (or null if no experience).

    # Evidence prompting requirements:
    # 1) Every skills_normalized item must include evidence from resume text.
    # 2) Every implied_skills item must include inferred_from with concise source phrase.
    # 3) Domain choice should be supported by role/company/responsibilities evidence in responsibilities or parsing_warnings when uncertain.
    # 4) Confidence score must reflect extraction certainty and evidence quality.
    # 5) For proficiency, infer from evidence strength:
    #    Beginner, Intermediate, Advanced, Expert only.

    # Business rules:
    # 1) parse_status = "SUCCESS" if extraction is reasonably structured, else "FAILED".
    # 2) Use candidate_status "PENDING" when confidence_score >= 0.7, otherwise "MANUAL_REVIEW".
    # 3) If unknown domain, use "Other".
    # 4) Keep dates in YYYY-MM or YYYY when possible.
    # 5) duplicate_pre_check must use:
    #    status in ["NO_MATCH", "POTENTIAL_MATCH"], checked_fields ["email","phone"], potential_matches [].
    # 6) storage must use:
    #    table="candidates_staging", candidate_status in ["PENDING","DUPLICATE_REVIEW","MANUAL_REVIEW"],
    #    awaiting="POC 2 - Deep Duplicate Analysis".
    # 7) If exact duplicate evidence is unavailable from input text alone, use duplicate_pre_check.status="NO_MATCH".

    # Self-check before final output:
    # 1) Validate all required top-level keys exist.
    # 2) Validate list/dict field types exactly.
    # 3) Ensure projects use "name" key.
    # 4) Ensure no markdown fences or commentary.
    # 5) Ensure confidence_score is 0.0 to 1.0.
    # 6) Ensure parse_status uses only SUCCESS/FAILED.
    # 7) Ensure all domains are from allowed list.

    # Resume text:
    # {safe_text}
    # """

    def _build_processing_prompt(
        self, raw_text: str, candidate_id: str, file_name: str
    ) -> str:
        safe_text = raw_text[:120000]

        return f"""
    You are an advanced Resume Intelligence Engine used in SmartRecruitz POC1.

    Your job is to convert messy, unstructured resume text into structured candidate data.

    The input may come from:
    • PDF
    • DOCX
    • TXT
    • OCR extracted text
    • scanned resumes
    • multi-column resumes
    • table-based resumes

    The text may contain:
    • broken lines
    • grammar mistakes
    • OCR errors
    • merged words
    • inconsistent formatting
    • missing section headers

    You must reconstruct meaning intelligently.

    -----------------------------------------------------

    Candidate ID: {candidate_id}
    File Name: {file_name}

    Allowed domain values: {", ".join(self.valid_domains)}
    Allowed proficiency values: Beginner, Intermediate, Advanced, Expert

    -----------------------------------------------------
    GENERAL RESUME UNDERSTANDING RULES

    Resumes can appear in many layouts:

    • multi-column layouts
    • table structures
    • bullet lists
    • compressed paragraphs
    • inconsistent ordering

    Section headers may vary.

    Example variations:

    Education:
    Education
    Academic Background
    Qualifications
    Academic Details

    Experience:
    Experience
    Work History
    Professional Experience
    Employment
    Career History

    Skills:
    Skills
    Technical Skills
    Core Competencies
    Technologies
    Tools

    Projects:
    Projects
    Personal Projects
    Academic Projects
    Research Work

    Certifications:
    Certifications
    Courses
    Professional Training
    Licenses

    You must detect these sections even if headers differ.

    -----------------------------------------------------

    PERSONAL INFORMATION EXTRACTION

    Always analyze the first 15 lines.

    Extract if present:
    full_name
    email
    phone
    location
    linkedin
    github
    portfolio

    Rules:

    • If first line contains 2–4 capitalized words → treat as full_name.
    • Emails may appear anywhere in first section.
    • Phone numbers may include country codes.
    • Location may appear as city or city/state.
    • LinkedIn may appear as text or URL.

    -----------------------------------------------------

    EXPERIENCE EXTRACTION

    Identify work experience entries.

    Each entry usually contains:

    company
    title
    start_date
    end_date
    responsibilities

    Date formats may include:

    Jan 2022 – Mar 2024
    2021 – Present
    03/2020 – 12/2023
    April 2023 – Current

    If role is ongoing:

    is_current = true
    end_date = "Present"

    Infer duration_months when possible.

    Responsibilities should summarize key tasks.

    -----------------------------------------------------

    EDUCATION EXTRACTION

    Education may appear in different formats.

    Examples:

    Bachelor of Technology – Computer Science
    B.Tech CSE
    Bachelor of Engineering (Mechanical)

    Table example:

    Degree | Institute | Year | CGPA

    Recognize abbreviations:

    B.Tech → Bachelor of Technology
    B.E → Bachelor of Engineering
    M.Tech → Master of Technology
    BSc → Bachelor of Science
    MSc → Master of Science

    Example parsing:

    "B.Tech CSE"

    degree = Bachelor of Technology
    field = Computer Science Engineering

    If specialization appears inside brackets:

    Bachelor of Technology – Computer Science (AI & ML)

    degree = Bachelor of Technology
    field = Computer Science (AI & ML)

    Extract:

    institution
    degree
    field
    graduation_year
    gpa

    -----------------------------------------------------

    SKILLS EXTRACTION

    Skills may appear in:

    • skills section
    • projects
    • experience descriptions

    Normalize variations.

    Examples:

    Python(OOP) → Python
    Postgresql → PostgreSQL
    LLMS → Large Language Models

    -----------------------------------------------------

    SKILL EVIDENCE GENERATION

    Evidence must describe how the candidate used the skill.

    Evidence sources:

    • experience
    • projects
    • responsibilities

    Good example:

    "Developed backend APIs using Python and FastAPI while building AI-driven automation workflows."

    Bad example:

    "Version Control : GitHub"

    If candidate is fresher, derive evidence from projects.

    Use action verbs:

    Developed
    Built
    Implemented
    Designed
    Applied

    -----------------------------------------------------

    DOMAIN CLASSIFICATION

    Determine domain using:

    • job titles
    • technologies
    • responsibilities

    Examples:

    Python + Machine Learning → AI/ML
    FastAPI + APIs → Software Engineering
    Warehouse operations → Logistics

    Choose best matching domain.

    -----------------------------------------------------

    OUTPUT STRUCTURE RULES

    Return JSON only.

    Top-level keys required exactly:

    candidate_id
    parse_status
    confidence_score
    personal_info
    experience
    education
    skills_raw
    skills_normalized
    implied_skills
    total_experience_years
    primary_domain
    domain_wise_experience
    certifications
    projects
    languages
    extra_data
    parsing_warnings
    duplicate_pre_check
    storage
    raw_text

    -----------------------------------------------------

    Critical output type rules:

    1) JSON only
    2) skills_raw must be array of strings
    3) skills_normalized objects:

    {{standard_name, original_terms[], proficiency, evidence, years_experience}}

    4) implied_skills objects:

    {{skill, inferred_from, confidence}}

    5) projects must use key "name"
       {{name, description, technologies[], domain, url, duration}}

    6) duplicate_pre_check must be object

    7) missing fields → null or empty arrays, but preserve valid types.

    8) extra_data must contain:

    awards[]
    publications[]
    links[]
    others[]

    Official required-field rules:
    1) personal_info.full_name is required when reasonably inferable.
    2) At least one of: experience[] not empty OR education[] not empty.
    3) If both are unavailable, set parse_status="FAILED" and add clear parsing_warnings.

    Experience extraction requirements (strict):
    1) experience[] objects must include at least company and title.
    2) Use keys exactly: company, title, start_date, end_date, duration_months, domain, responsibilities, is_current, location
    3) Normalize end_date to "Present" for current job.
    4) If dates are present, compute duration_months (positive integer).
    5) If fresher/no jobs, set experience=[] and total_experience_years=0.
    6) Prefer chronology and avoid overlap errors; add warnings in parsing_warnings if uncertain.
    7) domain per experience must be one allowed domain, else "Other".
    8) Validate no impossible timelines (future end dates, negative durations). If uncertain, keep dates but add warning.
    9) Compute total_experience_years from validated durations.
    10) Compute domain_wise_experience with fields:
        domain, experience_years, experience_months, companies[], roles[]
    11) Ensure sum(domain_wise_experience.experience_months) ~= total experience months; if mismatch, fix and warn.
    12) primary_domain must be domain with highest validated months (or null if no experience).

    Evidence prompting requirements:
    1) Every skills_normalized item must include evidence from resume text.
    2) Every implied_skills item must include inferred_from with concise source phrase.
    3) Domain choice should be supported by role/company/responsibilities evidence in responsibilities or parsing_warnings when uncertain.
    4) Confidence score must reflect extraction certainty and evidence quality.
    5) For proficiency, infer from evidence strength:
    Beginner, Intermediate, Advanced, Expert only
    -----------------------------------------------------

    BUSINESS RULES

    parse_status = SUCCESS when extraction structured.

    candidate_status = PENDING when confidence >= 0.7
    otherwise MANUAL_REVIEW.

    duplicate_pre_check:

    status = NO_MATCH or POTENTIAL_MATCH
    checked_fields = ["email","phone"]

    storage:

    table="candidates_staging"
    awaiting="POC 2 - Deep Duplicate Analysis"

    -----------------------------------------------------

    SELF VALIDATION

    Before returning output ensure:

    • all keys exist
    • JSON valid
    • correct array/dict types
    • projects use "name"
    • no markdown
    • confidence_score between 0.0 and 1.0
    • parse_status only SUCCESS or FAILED

    -----------------------------------------------------

    Resume text:
    {safe_text}
    """

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if code_block:
            response_text = code_block.group(1)

        json_blob = re.search(r"({[\s\S]*})", response_text)
        if json_blob:
            response_text = json_blob.group(1)

        return json.loads(response_text)

    def _validate_and_enhance(
        self,
        data: Dict[str, Any],
        candidate_id: str,
        raw_text: str,
    ) -> POC1Output:
        data = self._coerce_schema(data)
        data["candidate_id"] = data.get("candidate_id") or candidate_id

        confidence = float(data.get("confidence_score", 0.0) or 0.0)
        confidence = min(1.0, max(0.0, confidence))
        data["confidence_score"] = confidence
        data["parse_status"] = "SUCCESS" if confidence >= 0.5 else "FAILED"

        for exp in data.get("experience", []):
            domain = exp.get("domain")
            if domain and domain not in self.valid_domains:
                exp["domain"] = "Other"

        for domain_exp in data.get("domain_wise_experience", []):
            domain = domain_exp.get("domain")
            if domain and domain not in self.valid_domains:
                domain_exp["domain"] = "Other"

        if (
            data.get("primary_domain")
            and data["primary_domain"] not in self.valid_domains
        ):
            data["primary_domain"] = "Other"

        if not data.get("total_experience_years"):
            total_months = sum(
                int(item.get("duration_months") or 0)
                for item in data.get("experience", [])
            )
            data["total_experience_years"] = (
                round(total_months / 12.0, 1) if total_months else 0.0
            )

        storage = data.get("storage") or {}
        storage["table"] = "candidates_staging"
        storage["candidate_status"] = (
            "PENDING" if confidence >= 0.7 else "MANUAL_REVIEW"
        )
        storage["awaiting"] = (
            storage.get("awaiting") or "POC 2 - Deep Duplicate Analysis"
        )
        data["storage"] = storage

        if "raw_text" not in data or not data["raw_text"]:
            data["raw_text"] = raw_text[:1000]

        try:
            return POC1Output(**data)
        except ValidationError:
            repaired = self._repair_validation_sensitive_fields(data)
            return POC1Output(**repaired)

    def _coerce_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize common Gemini output shape errors into the expected schema."""
        data = dict(data or {})

        if not isinstance(data.get("personal_info"), dict):
            data["personal_info"] = {}

        if data.get("duplicate_pre_check") is None or not isinstance(
            data.get("duplicate_pre_check"), dict
        ):
            data["duplicate_pre_check"] = {
                "status": "NO_MATCH",
                "checked_fields": ["email", "phone"],
                "potential_matches": [],
            }

        data["skills_raw"] = self._coerce_string_list(data.get("skills_raw"))
        data["parsing_warnings"] = self._coerce_string_list(
            data.get("parsing_warnings")
        )

        if not isinstance(data.get("experience"), list):
            data["experience"] = []
        if not isinstance(data.get("education"), list):
            data["education"] = []
        if not isinstance(data.get("certifications"), list):
            data["certifications"] = []
        if not isinstance(data.get("projects"), list):
            data["projects"] = []
        if not isinstance(data.get("languages"), list):
            data["languages"] = []
        if not isinstance(data.get("domain_wise_experience"), list):
            data["domain_wise_experience"] = []

        data["skills_normalized"] = self._coerce_skills_normalized(
            data.get("skills_normalized")
        )
        data["implied_skills"] = self._coerce_implied_skills(data.get("implied_skills"))
        data["experience"] = [
            self._coerce_experience_item(item)
            for item in data.get("experience", [])
            if isinstance(item, dict)
        ]
        data["education"] = [
            self._coerce_education_item(item)
            for item in data.get("education", [])
            if isinstance(item, dict)
        ]
        data["certifications"] = [
            self._coerce_certification_item(item)
            for item in data.get("certifications", [])
            if isinstance(item, dict)
        ]
        data["projects"] = [
            self._coerce_project_item(item)
            for item in data.get("projects", [])
            if isinstance(item, dict)
        ]
        data["languages"] = [
            self._coerce_language_item(item)
            for item in data.get("languages", [])
            if isinstance(item, dict)
        ]
        data["domain_wise_experience"] = [
            self._coerce_domain_exp_item(item)
            for item in data.get("domain_wise_experience", [])
            if isinstance(item, dict)
        ]

        if not isinstance(data.get("extra_data"), dict):
            data["extra_data"] = {
                "awards": [],
                "publications": [],
                "links": [],
                "others": [],
            }
        else:
            extra = data["extra_data"]
            extra["awards"] = self._coerce_dict_list(extra.get("awards"))
            extra["publications"] = self._coerce_dict_list(extra.get("publications"))
            extra["links"] = self._coerce_string_list(extra.get("links"))
            extra["others"] = self._coerce_dict_list(extra.get("others"))

        return data

    def _coerce_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            # Split by newline, comma, or bullets.
            parts = re.split(r"[\n,;|]+|•|●|➢", value)
            return [p.strip() for p in parts if p and p.strip()]
        return [str(value).strip()] if str(value).strip() else []

    def _coerce_skills_normalized(self, value: Any) -> list[dict]:
        out: list[dict] = []
        if value is None:
            return out
        if isinstance(value, str):
            value = self._coerce_string_list(value)
        if not isinstance(value, list):
            return out

        for item in value:
            if isinstance(item, dict):
                name = str(item.get("standard_name") or "").strip()
                if not name:
                    continue
                proficiency = item.get("proficiency") or "Intermediate"
                if proficiency not in {
                    "Beginner",
                    "Intermediate",
                    "Advanced",
                    "Expert",
                }:
                    proficiency = "Intermediate"
                out.append(
                    {
                        "standard_name": name,
                        "original_terms": self._coerce_string_list(
                            item.get("original_terms") or [name]
                        ),
                        "proficiency": proficiency,
                        "evidence": str(item.get("evidence") or "Mentioned in resume"),
                        "years_experience": self._to_float(
                            item.get("years_experience")
                        ),
                    }
                )
            elif isinstance(item, str) and item.strip():
                name = item.strip()
                out.append(
                    {
                        "standard_name": name,
                        "original_terms": [name],
                        "proficiency": "Intermediate",
                        "evidence": "Mentioned in resume",
                        "years_experience": None,
                    }
                )
        return out

    def _coerce_implied_skills(self, value: Any) -> list[dict]:
        out: list[dict] = []
        if value is None:
            return out
        if isinstance(value, str):
            value = self._coerce_string_list(value)
        if not isinstance(value, list):
            return out

        for item in value:
            if isinstance(item, dict):
                skill = str(item.get("skill") or "").strip()
                if not skill:
                    continue
                confidence = self._to_float(item.get("confidence"))
                out.append(
                    {
                        "skill": skill,
                        "inferred_from": str(
                            item.get("inferred_from") or "Inferred from resume context"
                        ),
                        "confidence": min(
                            1.0, max(0.0, confidence if confidence is not None else 0.6)
                        ),
                    }
                )
            elif isinstance(item, str) and item.strip():
                out.append(
                    {
                        "skill": item.strip(),
                        "inferred_from": "Inferred from resume context",
                        "confidence": 0.6,
                    }
                )
        return out

    def _coerce_education_item(self, item: dict) -> dict:
        if not item.get("institution"):
            item["institution"] = (
                item.get("college")
                or item.get("college_name")
                or item.get("school")
                or "Unknown Institution"
            )
        gpa = item.get("gpa")
        item["gpa"] = self._to_float(gpa)
        return item

    def _coerce_experience_item(self, item: dict) -> dict:
        company = (
            item.get("company") or item.get("organization") or item.get("employer")
        )
        title = item.get("title") or item.get("role") or item.get("designation")
        item["company"] = str(company or "Unknown Company")
        item["title"] = str(title or "Unknown Role")
        if item.get("domain") and item.get("domain") not in self.valid_domains:
            item["domain"] = "Other"
        item["responsibilities"] = self._coerce_string_list(
            item.get("responsibilities")
        )
        return item

    def _coerce_certification_item(self, item: dict) -> dict:
        name = item.get("name") or item.get("title") or item.get("certification")
        item["name"] = str(name or "Unnamed Certification")
        item["year"] = int(self._to_float(item.get("year")) or 0) or None
        return item

    def _coerce_project_item(self, item: dict) -> dict:
        name = item.get("name") or item.get("title")
        description = item.get("description") or item.get("summary")
        item["name"] = str(name or "Untitled Project")
        item["description"] = str(
            description or "Project details extracted from resume"
        )
        item["technologies"] = self._coerce_string_list(
            item.get("technologies") or item.get("tech_stack")
        )
        if item.get("domain") and item.get("domain") not in self.valid_domains:
            item["domain"] = None
        return item

    def _coerce_language_item(self, item: dict) -> dict:
        language = item.get("language") or item.get("name")
        item["language"] = str(language or "Unknown")
        item["proficiency"] = str(item.get("proficiency") or "Professional")
        return item

    def _coerce_domain_exp_item(self, item: dict) -> dict:
        months = item.get("experience_months")
        years = self._to_float(item.get("experience_years")) or 0.0
        if months is None:
            months = int(round(years * 12))
        item["experience_months"] = int(self._to_float(months) or 0)
        item["experience_years"] = (
            round((item["experience_months"] / 12.0), 1) if years == 0 else years
        )
        item["companies"] = self._coerce_string_list(item.get("companies"))
        item["roles"] = self._coerce_string_list(item.get("roles"))
        return item

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return None
        return None

    def _coerce_dict_list(self, value: Any) -> list[dict]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            text = value.strip()
            return [{"value": text}] if text else []
        if not isinstance(value, list):
            return []

        output: list[dict] = []
        for item in value:
            if isinstance(item, dict):
                output.append(item)
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    output.append({"value": text})
        return output

    def _repair_validation_sensitive_fields(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Second-pass hardening: remove risky sections instead of failing the whole parse."""
        safe = dict(data)
        for key in (
            "skills_normalized",
            "implied_skills",
            "education",
            "domain_wise_experience",
            "experience",
            "certifications",
            "projects",
            "languages",
        ):
            if not isinstance(safe.get(key), list):
                safe[key] = []
        safe["experience"] = [
            self._coerce_experience_item(x)
            for x in safe["experience"]
            if isinstance(x, dict)
        ]
        safe["education"] = [
            self._coerce_education_item(x)
            for x in safe["education"]
            if isinstance(x, dict)
        ]
        safe["certifications"] = [
            self._coerce_certification_item(x)
            for x in safe["certifications"]
            if isinstance(x, dict)
        ]
        safe["projects"] = [
            self._coerce_project_item(x)
            for x in safe["projects"]
            if isinstance(x, dict)
        ]
        safe["languages"] = [
            self._coerce_language_item(x)
            for x in safe["languages"]
            if isinstance(x, dict)
        ]

        if not isinstance(safe.get("personal_info"), dict):
            safe["personal_info"] = {}
        if not isinstance(safe.get("storage"), dict):
            safe["storage"] = {
                "table": "candidates_staging",
                "candidate_status": "MANUAL_REVIEW",
                "awaiting": "Manual review required",
            }
        if not isinstance(safe.get("duplicate_pre_check"), dict):
            safe["duplicate_pre_check"] = {
                "status": "NO_MATCH",
                "checked_fields": ["email", "phone"],
                "potential_matches": [],
            }
        return safe

    def _create_fallback_output(
        self, raw_text: str, candidate_id: str, error: str
    ) -> POC1Output:
        return POC1Output(
            candidate_id=candidate_id,
            parse_status=ParseStatus.FAILED,
            confidence_score=0.0,
            personal_info=PersonalInfo(),
            parsing_warnings=[error],
            storage=StorageInfo(
                table="candidates_staging",
                candidate_status="MANUAL_REVIEW",
                awaiting="Manual review required",
            ),
            raw_text=raw_text[:1000],
        )
