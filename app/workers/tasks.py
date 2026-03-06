import logging
import platform
import time
from datetime import datetime, timezone
from typing import Optional

from celery import Celery
from celery.signals import worker_ready
import redis

from app.config import settings
from app.services.file_processor import FileProcessor

IS_WINDOWS = platform.system().lower().startswith("win")

# Initialize Celery
celery_app = Celery(
    "resume_parser",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    # Soft timeouts rely on SIGUSR1 and are unsupported on Windows.
    task_soft_time_limit=None if IS_WINDOWS else 25 * 60,
    task_default_queue=settings.CELERY_QUEUE,
    # Prefork on Windows can crash in task trace initialization.
    worker_pool="solo" if IS_WINDOWS else "prefork",
    task_routes={
        "process_resume": {"queue": settings.CELERY_QUEUE},
        "process_resume_v2": {"queue": settings.CELERY_QUEUE},
    },
)

if IS_WINDOWS:
    celery_app.conf.worker_concurrency = 1

logger = logging.getLogger(__name__)

# Process-local service instance for this worker process.
resume_service = None
NON_RETRYABLE_EXCEPTIONS = (
    TypeError,
    ValueError,
    RuntimeError,
    FileNotFoundError,
    NotImplementedError,
)


@worker_ready.connect
def _log_worker_source(**kwargs):
    logger.info(f"Celery worker loaded task module from {__file__}")
    logger.info(
        f"Worker runtime config: redis={settings.REDIS_URL} broker={settings.CELERY_BROKER_URL} queue={settings.CELERY_QUEUE} pool={celery_app.conf.worker_pool}"
    )


def _get_resume_service():
    """Lazily create a service instance in the worker process."""
    global resume_service
    if resume_service is None:
        from app.services.resume_service import ResumeService

        resume_service = ResumeService()
        logger.info(f"Initialized ResumeService in worker from {__file__}")
    return resume_service


def _mark_job_failed_direct(job_id: str, error: str):
    """Best-effort Redis update for early failures before service init."""
    try:
        client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        key = f"resume_job:{job_id}"
        raw = client.get(key)
        if not raw:
            return
        from app.models.job import JobResult

        job = JobResult.model_validate_json(raw)
        job.status = "failed"
        job.error = error
        job.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        client.set(key, job.model_dump_json())
    except Exception as ex:
        logger.error(f"Failed to mark job failed in Redis: job_id={job_id}, error={ex}")


@celery_app.task(bind=True, name="process_resume_v2")
def process_resume_task(
    self, job_id: str, file_path: str, filename: str, candidate_id: Optional[str] = None
):
    """
    Background task to process resume and extract text
    """
    logger.info(f"Starting resume processing: job_id={job_id}, file={filename}")

    try:
        service = _get_resume_service()

        # Update status to processing
        service.update_job_result(job_id, "processing")

        # Extract text using file processor
        processor = FileProcessor()
        start_time = time.time()

        # Determine file type from extension
        file_type = filename.split(".")[-1].lower() if "." in filename else None

        # Extract text
        raw_text = processor.extract_text(file_path, file_type)

        extraction_time = time.time() - start_time

        # Create extraction result
        result = {
            "job_id": job_id,
            "candidate_id": candidate_id,
            "raw_text": raw_text,
            "file_name": filename,
            "file_type": file_type,
            "text_length": len(raw_text),
            "extraction_time": extraction_time,
        }

        # Update job with result
        service.update_job_result(job_id, "completed", raw_text=raw_text)

        logger.info(
            f"Completed resume processing: job_id={job_id}, time={extraction_time:.2f}s"
        )
        return result

    except Exception as e:
        err = str(e)
        logger.error(f"Failed to process resume: job_id={job_id}, error={err}")

        max_retries = self.max_retries if self.max_retries is not None else 3
        will_retry = (
            not isinstance(e, NON_RETRYABLE_EXCEPTIONS)
            and self.request.retries < max_retries
        )

        if will_retry:
            retry_msg = f"{err} (retry {self.request.retries + 1}/{max_retries})"
            try:
                # Keep job non-terminal so temporary files are preserved for retry.
                service.update_job_result(job_id, "pending", error=retry_msg)
            except Exception:
                logger.exception(
                    f"Failed to update pending status before retry: job_id={job_id}"
                )
            raise self.retry(exc=e, countdown=60, max_retries=max_retries)

        try:
            service.update_job_result(job_id, "failed", error=err)
        except Exception:
            _mark_job_failed_direct(job_id, err)
        raise
