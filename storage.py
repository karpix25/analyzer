import json
from typing import Any, List, Optional

from redis.asyncio import Redis

from config import REDIS_QUEUE_NAME, REDIS_TASK_KEY_PREFIX, REDIS_TASK_SET_KEY, REDIS_URL
from schemas import ProcessingJob, TaskInfo

redis_client: Optional[Redis] = None


def _task_key(task_id: str) -> str:
    return f"{REDIS_TASK_KEY_PREFIX}{task_id}"


async def init_redis() -> None:
    """Создает подключение к Redis и проверяет доступность."""
    global redis_client
    redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
    await redis_client.ping()


async def close_redis() -> None:
    global redis_client
    if redis_client:
        await redis_client.close()
    redis_client = None


async def _get_redis() -> Redis:
    if redis_client is None:
        raise RuntimeError("Redis client is not initialized")
    return redis_client


async def save_task_info(task_info: TaskInfo) -> None:
    redis = await _get_redis()
    await redis.set(_task_key(task_info.task_id), task_info.json())
    await redis.sadd(REDIS_TASK_SET_KEY, task_info.task_id)


async def get_task_info(task_id: str) -> Optional[TaskInfo]:
    redis = await _get_redis()
    raw = await redis.get(_task_key(task_id))
    if not raw:
        return None
    return TaskInfo.parse_raw(raw)


async def update_task_fields(task_id: str, **fields: Any) -> Optional[TaskInfo]:
    task = await get_task_info(task_id)
    if not task:
        return None
    for key, value in fields.items():
        setattr(task, key, value)
    await save_task_info(task)
    return task


async def list_all_tasks() -> List[TaskInfo]:
    redis = await _get_redis()
    ids = await redis.smembers(REDIS_TASK_SET_KEY)
    if not ids:
        return []
    keys = [_task_key(tid) for tid in ids]
    raw_items = await redis.mget(*keys)
    tasks: List[TaskInfo] = []
    for raw in raw_items:
        if raw:
            tasks.append(TaskInfo.parse_raw(raw))
    return tasks


async def enqueue_job(job: ProcessingJob) -> None:
    redis = await _get_redis()
    await redis.lpush(REDIS_QUEUE_NAME, json.dumps(job.to_payload()))


async def fetch_job() -> Optional[ProcessingJob]:
    redis = await _get_redis()
    item = await redis.brpop(REDIS_QUEUE_NAME, timeout=5)
    if not item:
        return None
    _, payload = item
    return ProcessingJob.from_payload(json.loads(payload))
