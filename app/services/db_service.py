from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.database import Base, CandidateStaging
from app.models.poc1_models import POC1Output, StorageInfo

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

    async def quick_duplicate_check(
        self,
        email: Optional[str],
        phone: Optional[str],
    ) -> Tuple[bool, List[str]]:
        """
        Exact-match duplicate pre-check by email OR phone.
        Returns (has_duplicate, matched_candidate_ids).
        """
        if not email and not phone:
            return False, []

        conditions = []
        if email:
            conditions.append(
                CandidateStaging.profile_json["personal_info"]["email"].as_string() == email
            )
        if phone:
            conditions.append(
                CandidateStaging.profile_json["personal_info"]["phone"].as_string() == phone
            )

        async with self.async_session() as session:
            result = await session.execute(
                select(CandidateStaging.candidate_id).where(or_(*conditions))
            )
            matched_ids = [row[0] for row in result.all()]
            return bool(matched_ids), matched_ids

    async def save_candidate_profile(
        self,
        profile: POC1Output,
        job_id: str,
        raw_text: Optional[str] = None,
    ) -> str:
        email = str(profile.personal_info.email) if profile.personal_info and profile.personal_info.email else None
        phone = profile.personal_info.phone if profile.personal_info else None
        has_duplicate, matched_ids = await self.quick_duplicate_check(email=email, phone=phone)

        candidate_status = "DUPLICATE_REVIEW" if has_duplicate else "PENDING"
        if profile.storage:
            profile.storage.candidate_status = candidate_status
        else:
            profile.storage = StorageInfo(candidate_status=candidate_status)
        if profile.duplicate_pre_check:
            if has_duplicate:
                profile.duplicate_pre_check.status = "POTENTIAL_MATCH"
                profile.duplicate_pre_check.potential_matches = matched_ids
            else:
                profile.duplicate_pre_check.status = "NO_MATCH"
                profile.duplicate_pre_check.potential_matches = []

        async with self.async_session() as session:
            candidate = CandidateStaging(
                candidate_id=profile.candidate_id,
                job_id=job_id,
                profile_json=profile.model_dump(),
                raw_text=raw_text,
                status=candidate_status,
                confidence_score=profile.confidence_score,
                parse_status=profile.parse_status.value if hasattr(profile.parse_status, "value") else str(profile.parse_status),
                parsing_warnings=profile.parsing_warnings,
            )
            session.add(candidate)
            await session.commit()
            logger.info(
                "Saved candidate %s with status=%s (email=%s phone=%s matches=%s)",
                profile.candidate_id,
                candidate_status,
                email,
                phone,
                matched_ids,
            )
            return candidate.id

    async def get_candidate_by_id(self, candidate_id: str) -> Optional[CandidateStaging]:
        async with self.async_session() as session:
            result = await session.execute(
                select(CandidateStaging).where(CandidateStaging.candidate_id == candidate_id)
            )
            return result.scalar_one_or_none()

    async def get_candidate_by_email_or_phone(
        self, email: Optional[str], phone: Optional[str]
    ) -> List[CandidateStaging]:
        if not email and not phone:
            return []

        conditions = []
        if email:
            conditions.append(
                CandidateStaging.profile_json["personal_info"]["email"].as_string() == email
            )
        if phone:
            conditions.append(
                CandidateStaging.profile_json["personal_info"]["phone"].as_string() == phone
            )

        async with self.async_session() as session:
            result = await session.execute(
                select(CandidateStaging).where(or_(*conditions))
            )
            return result.scalars().all()

    async def update_candidate_status(self, candidate_id: str, status: str) -> None:
        async with self.async_session() as session:
            await session.execute(
                update(CandidateStaging)
                .where(CandidateStaging.candidate_id == candidate_id)
                .values(status=status, updated_at=datetime.utcnow())
            )
            await session.commit()
