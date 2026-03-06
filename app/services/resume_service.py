import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis

from app.config import settings
from app.models.job import JobResult
from app.workers.tasks import celery_app

logger = logging.getLogger(__name__)


class ResumeService:
    """Phase 2: Service Orchestration"""

    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def submit_resume(self, file_content: bytes, filename: str, candidate_id: Optional[str] = None) -> str:
        """
        Submit resume for processing
        Returns job_id
        """
        # Generate IDs
        job_id = str(uuid.uuid4())
        if not candidate_id:
            candidate_id = f"CAND-{uuid.uuid4().hex[:8].upper()}"

        # Save file temporarily
        file_path = await self._save_temp_file(file_content, filename, job_id)

        # Create job record
        job_result = JobResult(
            job_id=job_id,
            status="pending",
            candidate_id=candidate_id,
        )
        self._save_job(job_result)

        # Trigger background processing
        celery_app.send_task(
            "process_resume_v2",
            kwargs={
                "job_id": job_id,
                "file_path": file_path,
                "filename": filename,
                "candidate_id": candidate_id,
            },
            queue=settings.CELERY_QUEUE,
        )

        logger.info(f"Submitted resume for processing: job_id={job_id}, candidate_id={candidate_id}")
        return job_id

    async def _save_temp_file(self, content: bytes, filename: str, job_id: str) -> str:
        """Save uploaded file temporarily"""
        # Create job-specific directory
        job_dir = Path(settings.UPLOAD_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = job_dir / filename
        with open(file_path, "wb") as f:
            f.write(content)

        return str(file_path)

    def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get job status and result"""
        job_json = self.redis_client.get(self._job_key(job_id))
        if not job_json:
            return None
        return JobResult.model_validate_json(job_json)

    def update_job_result(
        self,
        job_id: str,
        status: str,
        raw_text: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Update job result (called by Celery task)"""
        job = self.get_job_status(job_id)
        if not job:
            logger.error(f"Job not found for update: {job_id}")
            return

        job.status = status
        if raw_text is not None:
            job.raw_text = raw_text
        if error is not None:
            job.error = error
        if status in ["completed", "failed"]:
            job.completed_at = datetime.utcnow()

        self._save_job(job)

        # Clean up temp files if completed/failed
        if status in ["completed", "failed"]:
            self._cleanup_temp_files(job_id)

    def _cleanup_temp_files(self, job_id: str):
        """Clean up temporary files"""
        job_dir = Path(settings.UPLOAD_DIR) / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up temp files for job {job_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup temp files for job {job_id}: {e}")

    def _job_key(self, job_id: str) -> str:
        return f"resume_job:{job_id}"

    def _save_job(self, job: JobResult):
        self.redis_client.set(self._job_key(job.job_id), job.model_dump_json())
