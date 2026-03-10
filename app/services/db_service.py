from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.database import Base, CandidateStaging
from app.models.poc1_models import POC1Output

logger = logging.getLogger(__name__)


class DatabaseService:
    """Persist structured POC1 profiles in PostgreSQL staging table."""

    def __init__(self):
        self.engine = create_async_engine(settings.DATABASE_URL, echo=False)
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_models(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_candidate_profile(
        self,
        profile: POC1Output,
        job_id: str,
        raw_text: Optional[str] = None,
    ) -> str:
        async with self.async_session() as session:
            candidate = CandidateStaging(
                candidate_id=profile.candidate_id,
                job_id=job_id,
                profile_json=profile.model_dump(),
                raw_text=raw_text,
                status=profile.storage.candidate_status if profile.storage else "PENDING",
                confidence_score=profile.confidence_score,
                parse_status=profile.parse_status.value if hasattr(profile.parse_status, "value") else str(profile.parse_status),
                parsing_warnings=profile.parsing_warnings,
            )
            session.add(candidate)
            await session.commit()
            return candidate.id

    async def get_candidate_by_id(self, candidate_id: str) -> Optional[CandidateStaging]:
        async with self.async_session() as session:
            result = await session.execute(
                select(CandidateStaging).where(CandidateStaging.candidate_id == candidate_id)
            )
            return result.scalar_one_or_none()

    async def update_candidate_status(self, candidate_id: str, status: str) -> None:
        async with self.async_session() as session:
            await session.execute(
                update(CandidateStaging)
                .where(CandidateStaging.candidate_id == candidate_id)
                .values(status=status, updated_at=datetime.utcnow())
            )
            await session.commit()
