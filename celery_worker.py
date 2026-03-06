#!/usr/bin/env python
import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

from app.workers.tasks import celery_app

if __name__ == "__main__":
    celery_app.start()
