from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class ParseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ProficiencyLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


class Domain(str, Enum):
    FINTECH = "FinTech"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-Commerce"
    ED_TECH = "EdTech"
    TRAVEL = "Travel"
    LOGISTICS = "Logistics"
    MANUFACTURING = "Manufacturing"
    RETAIL = "Retail"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    TELECOM = "Telecom"
    MEDIA = "Media"
    ENERGY = "Energy"
    AUTOMOTIVE = "Automotive"
    AEROSPACE = "Aerospace"
    GOVERNMENT = "Government"
    NONPROFIT = "Non-Profit"
    OTHER = "Other"


class DuplicatePreCheck(BaseModel):
    status: str = "NO_MATCH"
    checked_fields: List[str] = Field(default_factory=lambda: ["email", "phone"])
    potential_matches: List[str] = Field(default_factory=list)


class StorageInfo(BaseModel):
    table: str = "candidates_staging"
    candidate_status: str = Field(
        ...,
        description="PENDING, DUPLICATE_REVIEW, MANUAL_REVIEW",
    )
    awaiting: str = "POC 2 - Deep Duplicate Analysis"


class PersonalInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    current_title: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: Optional[str]) -> Optional[str]:
        if value:
            digits = re.sub(r"\D", "", value)
            if len(digits) < 10:
                raise ValueError("Phone number too short")
        return value


class Experience(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    domain: Optional[Domain] = None
    responsibilities: List[str] = Field(default_factory=list)
    is_current: bool = False
    location: Optional[str] = None

    @field_validator("end_date")
    @classmethod
    def normalize_end_date(cls, value: Optional[str]) -> Optional[str]:
        if value and value.lower() == "present":
            return "Present"
        return value


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field: Optional[str] = None
    graduation_year: Optional[int] = None
    graduation_date: Optional[str] = None
    gpa: Optional[float] = None
    achievements: List[str] = Field(default_factory=list)


class NormalizedSkill(BaseModel):
    standard_name: str
    original_terms: List[str]
    proficiency: ProficiencyLevel
    evidence: str
    years_experience: Optional[float] = None


class ImpliedSkill(BaseModel):
    skill: str
    inferred_from: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class DomainExperience(BaseModel):
    domain: Domain
    experience_years: float
    experience_months: int
    companies: List[str]
    roles: List[str]


class Certification(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None
    valid_until: Optional[str] = None
    url: Optional[str] = None


class Project(BaseModel):
    name: str
    description: str
    technologies: List[str] = Field(default_factory=list)
    domain: Optional[Domain] = None
    url: Optional[str] = None
    duration: Optional[str] = None


class Language(BaseModel):
    language: str
    proficiency: str


class ExtraData(BaseModel):
    awards: List[Dict[str, Any]] = Field(default_factory=list)
    publications: List[Dict[str, Any]] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    others: List[Dict[str, Any]] = Field(default_factory=list)


class POC1Output(BaseModel):
    candidate_id: str
    parse_status: ParseStatus
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    personal_info: PersonalInfo
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills_raw: List[str] = Field(default_factory=list)
    skills_normalized: List[NormalizedSkill] = Field(default_factory=list)
    implied_skills: List[ImpliedSkill] = Field(default_factory=list)
    total_experience_years: float = 0.0
    primary_domain: Optional[Domain] = None
    domain_wise_experience: List[DomainExperience] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    extra_data: ExtraData = Field(default_factory=ExtraData)
    parsing_warnings: List[str] = Field(default_factory=list)
    duplicate_pre_check: DuplicatePreCheck = Field(default_factory=DuplicatePreCheck)
    storage: Optional[StorageInfo] = None
    raw_text: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "candidate_id": "SR-2024-00123",
                "parse_status": "SUCCESS",
                "confidence_score": 0.94,
            }
        }
    }
