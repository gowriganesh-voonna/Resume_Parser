import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import resume
from app.config import settings
from app.services.resume_service import ResumeService

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
resume_service = ResumeService()

# Initialize routes
resume.init_routes(resume_service)

# Include routers
app.include_router(resume.router)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Resume Parser API", version=settings.APP_VERSION)
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(
        "Runtime config",
        redis_url=settings.REDIS_URL,
        broker=settings.CELERY_BROKER_URL,
        queue=settings.CELERY_QUEUE,
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Resume Parser API")


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "upload_dir_exists": True,
        "celery_broker": settings.CELERY_BROKER_URL,
    }
