import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoSubmitRequest(BaseModel):
    """Запрос на обработку видео по URL."""
    video_url: HttpUrl
    webhook_url: Optional[HttpUrl] = None
    callback_data: Optional[Dict[str, Any]] = None


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    callback_data: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None


@dataclass
class ProcessingJob:
    task_id: str
    source: str  # "upload" or "url"
    temp_path: Optional[str] = None
    video_url: Optional[str] = None
    webhook_url: Optional[str] = None
    callback_data: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "temp_path": self.temp_path,
            "video_url": self.video_url,
            "webhook_url": self.webhook_url,
            "callback_data": self.callback_data,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ProcessingJob":
        return cls(
            task_id=payload["task_id"],
            source=payload["source"],
            temp_path=payload.get("temp_path"),
            video_url=payload.get("video_url"),
            webhook_url=payload.get("webhook_url"),
            callback_data=payload.get("callback_data"),
        )


def new_task_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.utcnow().isoformat()
