import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.job import JobResponse, JobResult
from app.models.poc1_models import POC1Output
from app.services.layout_detector import LayoutDetector
from app.services.resume_service import ResumeService
from app.workers.tasks import celery_app

router = APIRouter(prefix="/resume", tags=["resume"])
logger = logging.getLogger(__name__)

# Service instance (will be set by main app)
resume_service: Optional[ResumeService] = None


def init_routes(service: ResumeService):
    """Initialize routes with service"""
    global resume_service
    resume_service = service


@router.post("/upload", response_model=JobResponse)
async def upload_resume(
    file: UploadFile = File(...),
    candidate_id: Optional[str] = None,
):
    """
    Upload a resume file for processing
    Returns job_id for tracking
    """
    if not resume_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    # Validate file type
    allowed_types = [".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg"]
    file_ext = f".{file.filename.split('.')[-1].lower()}" if "." in file.filename else ""

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(allowed_types)}",
        )

    try:
        # Read file content
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size should be less than or equal to {settings.MAX_FILE_SIZE // (1024 * 1024)}MB",
            )

        # Submit for processing
        job_id = await resume_service.submit_resume(
            file_content=content,
            filename=file.filename,
            candidate_id=candidate_id,
        )

        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Resume uploaded successfully. Processing started.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobResult)
async def get_job_status(job_id: str):
    """
    Get job status and extraction result
    """
    if not resume_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    job = resume_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.get("/{job_id}/text")
async def get_extracted_text(job_id: str):
    """
    Get just the extracted text for a completed job
    """
    if not resume_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    job = resume_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ["pending", "processing"]:
        raise HTTPException(status_code=202, detail="Processing in progress")

    if job.status == "failed":
        raise HTTPException(status_code=500, detail=f"Processing failed: {job.error}")

    if not job.raw_text:
        raise HTTPException(status_code=404, detail="No text extracted")

    return JSONResponse(
        content={
            "job_id": job.job_id,
            "candidate_id": job.candidate_id,
            "status": job.status,
            "text": job.raw_text,
            "text_length": len(job.raw_text),
        }
    )


@router.get("/{job_id}/profile", response_model=POC1Output)
async def get_candidate_profile(job_id: str):
    """
    Get the complete processed candidate profile.
    """
    if not resume_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    profile = resume_service.get_processed_profile(job_id)
    if not profile:
        job = resume_service.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(status_code=202, detail="Processing still in progress")
        raise HTTPException(status_code=404, detail="Profile not found")

    return profile


@router.get("/{job_id}/profile/json")
async def get_candidate_profile_json(job_id: str):
    """
    Get profile in raw JSON format.
    """
    if not resume_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    profile = resume_service.get_processed_profile(job_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return JSONResponse(content=profile.model_dump(mode="json"))


@router.post("/{job_id}/reprocess")
async def reprocess_with_ai(job_id: str):
    """
    Reprocess an existing job's raw text via AI.
    """
    if not resume_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    job = resume_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.raw_text:
        raise HTTPException(status_code=400, detail="No raw text available")

    celery_app.send_task(
        "reprocess_with_ai",
        kwargs={
            "job_id": job_id,
            "raw_text": job.raw_text,
            "candidate_id": job.candidate_id,
        },
        queue=settings.CELERY_QUEUE,
    )

    return {"status": "reprocessing_started", "job_id": job_id}


@router.post("/layout/analyze")
async def analyze_layout(file: UploadFile = File(...)):
    """
    Analyze layout of an uploaded PDF for extraction diagnostics.
    """
    file_ext = f".{file.filename.split('.')[-1].lower()}" if "." in file.filename else ""
    if file_ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported for layout analysis")

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size should be less than or equal to {settings.MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(content)
            temp_path = temp.name

        detector = LayoutDetector()
        layout = detector.analyze_document(temp_path)

        return JSONResponse(
            content={
                "file_name": file.filename,
                "layout_type": layout["layout_type"].value,
                "column_count": layout["column_count"],
                "has_tables": layout["has_tables"],
                "has_images": layout["has_images"],
                "pages": layout["pages"],
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Layout analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                logger.warning("Failed to remove temporary file used for layout analysis")
