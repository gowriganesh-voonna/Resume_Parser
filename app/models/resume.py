from pydantic import BaseModel


class ResumeMetadata(BaseModel):
    file_name: str
    file_type: str
    size_bytes: int
