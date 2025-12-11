import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env early so all modules can read env vars.
load_dotenv()

# Paths
RESULT_DIR = Path("./results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Public URLs
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://test-image-analyze.g44y6r.easypanel.host/").rstrip("/")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_QUEUE_NAME = os.getenv("REDIS_QUEUE", "video_tasks")
REDIS_TASK_SET_KEY = os.getenv("REDIS_TASK_SET_KEY", "video_tasks:ids")
REDIS_TASK_KEY_PREFIX = os.getenv("REDIS_TASK_KEY_PREFIX", "video_task:")

# Workers
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "3"))
