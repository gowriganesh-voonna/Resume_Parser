"""
Quick verification for exact duplicate pre-check logic (email OR phone).

Run:
    python -m scripts.test_duplicate_logic
"""

import asyncio
import uuid

from app.models.poc1_models import POC1Output, ParseStatus, PersonalInfo
from app.services.db_service import DatabaseService


def _profile(candidate_id: str, email: str, phone: str) -> POC1Output:
    return POC1Output(
        candidate_id=candidate_id,
        parse_status=ParseStatus.SUCCESS,
        confidence_score=0.95,
        personal_info=PersonalInfo(
            full_name="Test User",
            email=email,
            phone=phone,
        ),
    )


async def main() -> None:
    db = DatabaseService()
    await db.init_models()

    p1 = _profile(
        candidate_id=f"TEST-{uuid.uuid4().hex[:8].upper()}",
        email="dup-test@example.com",
        phone="+1-555-111-2222",
    )
    await db.save_candidate_profile(p1, job_id=f"JOB-{uuid.uuid4().hex[:8].upper()}")
    print("Case 1:", p1.storage.candidate_status, p1.duplicate_pre_check.status)

    p2 = _profile(
        candidate_id=f"TEST-{uuid.uuid4().hex[:8].upper()}",
        email="dup-test@example.com",  # same email
        phone="+1-555-333-4444",
    )
    await db.save_candidate_profile(p2, job_id=f"JOB-{uuid.uuid4().hex[:8].upper()}")
    print("Case 2:", p2.storage.candidate_status, p2.duplicate_pre_check.potential_matches)

    p3 = _profile(
        candidate_id=f"TEST-{uuid.uuid4().hex[:8].upper()}",
        email="new-email@example.com",
        phone="+1-555-111-2222",  # same phone as p1
    )
    await db.save_candidate_profile(p3, job_id=f"JOB-{uuid.uuid4().hex[:8].upper()}")
    print("Case 3:", p3.storage.candidate_status, p3.duplicate_pre_check.potential_matches)


if __name__ == "__main__":
    asyncio.run(main())
