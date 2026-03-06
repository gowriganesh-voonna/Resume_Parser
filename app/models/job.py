from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    status: str
    candidate_id: Optional[str] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class ExtractionResult(BaseModel):
    candidate_id: str
    raw_text: str
    file_name: str
    file_type: str
    text_length: int
    extraction_time: float
