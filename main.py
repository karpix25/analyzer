import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

from config import PUBLIC_BASE_URL, RESULT_DIR, USE_S3, WORKER_CONCURRENCY
from processing import crop_video_ffmpeg, estimate_crop_box, refine_crop_rect, sample_frames, select_best_frame
from schemas import ProcessingJob, TaskInfo, TaskStatus, VideoSubmitRequest, new_task_id, now_iso
from storage import (
    close_redis,
    enqueue_job,
    fetch_job,
    get_task_info,
    init_redis,
    list_all_tasks,
    save_task_info,
    update_task_fields,
)
from storage_s3 import s3_storage


# -----------------------------------------------------------------------------
# Logging: force stdout handler so logs видны в Docker/Easypanel
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),  # Also stderr for Easypanel
    ],
    force=True,
)
print("=" * 80, file=sys.stderr, flush=True)
print("VIDEO ANALYZER STARTING", file=sys.stderr, flush=True)
print("=" * 80, file=sys.stderr, flush=True)
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(name).setLevel(logging.INFO)
logger = logging.getLogger("app")

# ---------------------------------
# App & CORS
# ---------------------------------
app = FastAPI(title="Video Box Detector - Async")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background worker tasks
worker_tasks = []


def _resolve_base_url() -> str:
    """Возвращает базовый URL для файлов результата."""
    return PUBLIC_BASE_URL.rstrip("/")


# ---------------------------------
# Background workers
# ---------------------------------
async def worker_loop(worker_id: int) -> None:
    """Обрабатывает задания из очереди."""
    while True:
        try:
            job = await fetch_job()
            if job is None:
                continue
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[WORKER {worker_id}] Failed to fetch job: {e}")
            await asyncio.sleep(1.0)
            continue

        video_path: Optional[Path] = None

        try:
            logger.info(f"[WORKER {worker_id}] Picked task {job.task_id} ({job.source})")

            task_info = await get_task_info(job.task_id)
            if task_info is None:
                logger.warning(f"[WORKER {worker_id}] Task {job.task_id} not found in storage")
                continue

            await update_task_fields(job.task_id, status=TaskStatus.PROCESSING, error=None, result=None)

            if job.source == "url":
                if not job.video_url:
                    raise ValueError("Video URL is required for url job")

                logger.info(f"[TASK {job.task_id}] Starting download from URL")
                try:
                    video_path = await download_video_from_url(job.video_url)
                    logger.info(f"[TASK {job.task_id}] Download completed: {video_path}")
                except Exception as e:
                    error_msg = f"Download failed: {str(e)}"
                    logger.error(f"[TASK {job.task_id}] {error_msg}")
                    await update_task_fields(
                        job.task_id,
                        status=TaskStatus.FAILED,
                        error=error_msg,
                        result=None,
                        completed_at=now_iso(),
                    )
                    if job.webhook_url:
                        await send_webhook(job.webhook_url, job.task_id, None, job.callback_data, error=error_msg)
                    continue
            else:
                if not job.temp_path:
                    raise ValueError("Temp file is required for upload job")
                video_path = Path(job.temp_path)

            await process_video_task(job.task_id, video_path, job.webhook_url, job.callback_data)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_msg = f"Worker error: {type(e).__name__}: {str(e)}"
            logger.error(f"[WORKER {worker_id}] {error_msg}")
            task_info = await get_task_info(job.task_id)
            if task_info:
                await update_task_fields(
                    job.task_id,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                    result=None,
                    completed_at=now_iso(),
                )
                if job.webhook_url:
                    await send_webhook(job.webhook_url, job.task_id, None, job.callback_data, error=error_msg)

    logger.info(f"[WORKER {worker_id}] Stopped")


@app.on_event("startup")
async def app_startup() -> None:
    """Инициализация очереди и воркеров."""
    global worker_tasks
    await init_redis()
    worker_tasks = [asyncio.create_task(worker_loop(idx + 1)) for idx in range(max(1, WORKER_CONCURRENCY))]
    logger.info(f"[STARTUP] Created {len(worker_tasks)} worker(s)")


@app.on_event("shutdown")
async def app_shutdown() -> None:
    """Корректно останавливает воркеров и Redis."""
    global worker_tasks
    for task in worker_tasks:
        task.cancel()

    if worker_tasks:
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    worker_tasks = []
    await close_redis()


# ---------------------------------
# Фоновая обработка
# ---------------------------------
async def process_video_task(
    task_id: str,
    video_path: Path,
    webhook_url: Optional[str],
    callback_data: Optional[Dict[str, Any]],
):
    """Фоновая задача обработки видео."""
    logger.info(f"[TASK {task_id}] Started processing")

    await update_task_fields(task_id, status=TaskStatus.PROCESSING, error=None, result=None)

    try:
        frames = sample_frames(str(video_path), max_frames=20)

        if len(frames) < 2:
            raise Exception("Недостаточно кадров для анализа")

        bbox_rough, text_bottom, is_motion = estimate_crop_box(frames, task_id)
        motion_x, motion_y, motion_w, motion_h = bbox_rough

        # Calculate median frame to isolate static background and remove dynamic noise
        frames_arr = np.array(frames)
        median_frame = np.median(frames_arr, axis=0).astype(np.uint8)
        H, W = median_frame.shape[:2]
        
        if is_motion:
            # HYBRID APPROACH when motion detection succeeded:
            # - Use Motion Detection for HORIZONTAL bounds (x, width) - they are accurate
            # - Use Refine for VERTICAL bounds (y, height) - to trim top/bottom uniform strips
            # This prevents: 1) cutting video content (sand), 2) missing top bars (orange)
            
            # ВАЖНО: Работаем только с контентной областью (ниже text_bottom)
            # Берём ROI начиная от text_bottom до конца кадра
            content_start_y = text_bottom
            content_roi = median_frame[content_start_y:, :]  # Только контент, без текста
            content_h = H - content_start_y
            
            # Apply refine ТОЛЬКО к контентной области
            _, roi_y, _, roi_h = refine_crop_rect(
                content_roi, 0, 0, W, content_h,
                task_id=task_id,
                full_frame=median_frame,
                roi_offset_y=text_bottom
            )
            
            # Пересчитываем координаты в полный кадр
            refined_y = content_start_y + roi_y
            refined_h = roi_h
            
            # Combine: Motion X/W + Refined Y/H
            cx = motion_x
            cw = motion_w
            cy = refined_y
            ch = refined_h
        else:
            # Fallback: no motion detected, use refine on content area only
            # Работаем только с областью ниже text_bottom
            content_start_y = text_bottom
            content_roi = median_frame[content_start_y:, :]
            content_h = H - content_start_y
            
            # ИСПРАВЛЕНИЕ: Применяем refine ТОЛЬКО для высоты
            # Ширину берём из motion detection (она правильная)
            _, roi_y, _, roi_h = refine_crop_rect(
                content_roi, 0, 0, W, content_h,
                task_id=task_id,
                full_frame=median_frame,
                roi_offset_y=text_bottom
            )
            
            # Координаты: ширину берём из motion, высоту из refine
            cx = motion_x
            cw = motion_w
            cy = content_start_y + roi_y
            ch = roi_h
            
        bbox_clean = (cx, cy, cw, ch)
        bbox_rough = (motion_x, motion_y, motion_w, motion_h)  # For debug/logging

        best_frame, quality = select_best_frame(frames, bbox_clean)
        preview = best_frame[cy : cy + ch, cx : cx + cw]

        task_result_dir = RESULT_DIR / task_id

        cv2.imwrite(str(task_result_dir / "frame.jpg"), preview)

        H = frames[0].shape[0]
        if text_bottom > 50:
            text_frame = best_frame[0:text_bottom, :]
            cv2.imwrite(str(task_result_dir / "text_frame.jpg"), text_frame)

        video_crop, text_crop, clean_crop = crop_video_ffmpeg(video_path, task_id, text_bottom, bbox_rough, bbox_clean)

        # Upload to S3 if enabled
        if USE_S3 and s3_storage.enabled:
            logger.info(f"[TASK {task_id}] Uploading results to S3...")
            s3_urls = s3_storage.upload_task_results(task_id, task_result_dir)
            
            # Use S3 URLs if upload successful
            base_url = None  # Not needed for S3
            files = {
                "content_video": s3_urls.get("content_video"),
                "clean_video": s3_urls.get("clean_video"),
                "content_frame": s3_urls.get("content_frame"),
                "text_video": s3_urls.get("text_video"),
                "text_frame": s3_urls.get("text_frame"),
                "debug_frame": s3_urls.get("debug_frame"),
                "density_profile": s3_urls.get("density_profile"),
                "clean_crop_debug": s3_urls.get("clean_crop_debug"),
            }
            logger.info(f"[TASK {task_id}] S3 upload completed")
        else:
            # Use local URLs
            base_url = _resolve_base_url()
            files = {
                "content_video": f"{base_url}/result/{task_id}/video_crop.mp4",
                "clean_video": f"{base_url}/result/{task_id}/clean_crop.mp4" if clean_crop else None,
                "content_frame": f"{base_url}/result/{task_id}/frame.jpg",
                "text_video": f"{base_url}/result/{task_id}/text_crop.mp4" if text_crop else None,
                "text_frame": f"{base_url}/result/{task_id}/text_frame.jpg",
                "debug_frame": f"{base_url}/result/{task_id}/debug.jpg",
                "density_profile": f"{base_url}/result/{task_id}/density_profile.jpg",
                "clean_crop_debug": f"{base_url}/result/{task_id}/clean_crop_debug.jpg",
            }

        result = {
            "task_id": task_id,
            "box": {"x": motion_x, "y": motion_y, "w": motion_w, "h": motion_h},
            "clean_box": {"x": cx, "y": cy, "w": cw, "h": ch},
            "text_bottom": int(text_bottom),
            "score": round(float(quality), 4),
            "files": files,
        }

        await update_task_fields(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=now_iso(),
            result=result,
            error=None,
        )

        logger.info(f"[TASK {task_id}] Completed successfully")

        if webhook_url:
            await send_webhook(webhook_url, task_id, result, callback_data)

    except Exception as e:
        logger.error(f"[TASK {task_id}] Failed: {str(e)}")
        await update_task_fields(
            task_id,
            status=TaskStatus.FAILED,
            error=str(e),
            completed_at=now_iso(),
            result=None,
        )

        if webhook_url:
            await send_webhook(webhook_url, task_id, None, callback_data, error=str(e))

    finally:
        try:
            video_path.unlink()
        except Exception:
            pass


async def send_webhook(webhook_url: str, task_id: str, result: Optional[Dict], callback_data: Optional[Dict], error: Optional[str] = None):
    """Отправляет webhook с результатом."""
    if not webhook_url:
        logger.info(f"[WEBHOOK] Skipping for task {task_id} - no webhook_url provided")
        return

    payload = {
        "task_id": task_id,
        "status": "completed" if result else "failed",
        "callback_data": callback_data,
        "timestamp": now_iso(),
    }

    if result:
        payload["result"] = result
    else:
        payload["error"] = error

    logger.info(f"[WEBHOOK] Sending to {webhook_url} for task {task_id}")
    logger.debug(f"[WEBHOOK] Payload keys: {list(payload.keys())}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=30.0)
            logger.info(f"[WEBHOOK] Response status: {response.status_code}")
            logger.debug(f"[WEBHOOK] Response headers: {dict(response.headers)}")
            logger.debug(f"[WEBHOOK] Response body: {response.text[:200]}")
    except httpx.TimeoutException:
        logger.error(f"[WEBHOOK] Timeout sending to {webhook_url}")
    except httpx.RequestError as e:
        logger.error(f"[WEBHOOK] Request error: {type(e).__name__}: {str(e)}")
    except Exception as e:
        logger.error(f"[WEBHOOK] Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


async def download_video_from_url(url: str) -> Path:
    """Скачивает видео по URL."""
    parsed_url = urlparse(url)
    suffix = Path(unquote(parsed_url.path)).suffix.lower()
    allowed_suffixes = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}
    if suffix not in allowed_suffixes:
        suffix = ".mp4"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, timeout=60.0) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
    except Exception:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise

    return temp_path


# ---------------------------------
# API Endpoints
# ---------------------------------
@app.post("/detect/video")
async def detect_video_multipart(
    video: UploadFile = File(...),
    webhook_url: Optional[str] = None,
    callback_data: Optional[str] = None,
):
    """Асинхронная обработка видео (multipart/form-data)."""
    task_id = new_task_id()

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        content = await video.read()
        temp_file.write(content)
        temp_file.flush()
        temp_path = Path(temp_file.name)
    finally:
        temp_file.close()

    parsed_callback_data = None
    if callback_data:
        try:
            parsed_callback_data = json.loads(callback_data)
        except Exception:
            parsed_callback_data = {"data": callback_data}

    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=now_iso(),
        callback_data=parsed_callback_data,
        webhook_url=webhook_url,
    )
    await save_task_info(task_info)

    logger.info(f"[TASK {task_id}] Created with webhook_url: {webhook_url}")

    job = ProcessingJob(
        task_id=task_id,
        source="upload",
        temp_path=str(temp_path),
        webhook_url=webhook_url,
        callback_data=parsed_callback_data,
    )

    await enqueue_job(job)
    logger.info(f"[TASK {task_id}] Enqueued for processing (upload)")

    return JSONResponse(
        {
            "task_id": task_id,
            "status": "pending",
            "message": "Video processing started",
            "status_url": f"/task/{task_id}",
            "webhook_url": webhook_url,
        }
    )


@app.post("/detect/video/url")
async def detect_video_url(
    request: VideoSubmitRequest,
):
    """Асинхронная обработка видео по URL."""
    task_id = new_task_id()

    webhook_url_str = str(request.webhook_url) if request.webhook_url else None

    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=now_iso(),
        callback_data=request.callback_data,
        webhook_url=webhook_url_str,
    )
    await save_task_info(task_info)

    logger.info(f"[TASK {task_id}] Created with webhook_url: {webhook_url_str}")

    job = ProcessingJob(
        task_id=task_id,
        source="url",
        video_url=str(request.video_url),
        webhook_url=webhook_url_str,
        callback_data=request.callback_data,
    )

    await enqueue_job(job)
    logger.info(f"[TASK {task_id}] Enqueued for processing (url)")

    return JSONResponse(
        {
            "task_id": task_id,
            "status": "pending",
            "message": "Video download and processing started",
            "status_url": f"/task/{task_id}",
            "webhook_url": webhook_url_str,
        }
    )


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Получить статус задачи."""
    task_info = await get_task_info(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_info


@app.get("/result/{task_id}/{filename}")
async def get_result_file(task_id: str, filename: str):
    """Получить файл результата."""
    file_path = RESULT_DIR / task_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".jpg"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=filename,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "video-analyzer",
        "s3_enabled": USE_S3,
    }


@app.get("/tasks")
async def list_tasks():
    """Список всех задач."""
    try:
        tasks = await list_all_tasks()
        return {"tasks": tasks}
    except Exception as e:
        return {"error": str(e), "tasks": []}


@app.get("/debug/tasks")
async def debug_tasks():
    """Debug endpoint to check all task statuses."""
    try:
        tasks = await list_all_tasks()
        
        # Sort by created_at (newest first)
        tasks_sorted = sorted(
            tasks, 
            key=lambda t: t.created_at if hasattr(t, 'created_at') and t.created_at else "", 
            reverse=True
        )
        
        stats = {
            "total": len(tasks_sorted),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }
        
        task_details = []
        for task in tasks_sorted:
            # TaskInfo is a Pydantic model, use attributes directly
            status = task.status if hasattr(task, 'status') else "unknown"
            if status in stats:
                stats[status] += 1
            else:
                stats[status] = 1
            
            # Extract result details if available
            result_info = None
            if hasattr(task, 'result') and task.result:
                result_info = {
                    "files": task.result.get("files", {}),
                    "crop_info": {
                        "text_bottom": task.result.get("text_bottom"),
                        "content_bbox": task.result.get("box"),
                        "clean_bbox": task.result.get("clean_box"),
                    }
                }
            
            task_details.append({
                "task_id": task.task_id if hasattr(task, 'task_id') else None,
                "status": status,
                "created_at": task.created_at if hasattr(task, 'created_at') else None,
                "completed_at": task.completed_at if hasattr(task, 'completed_at') else None,
                "error": task.error if hasattr(task, 'error') else None,
                "result": result_info,
            })
        
        return {
            "stats": stats,
            "tasks": task_details,
        }
    except Exception as e:
        return {
            "error": str(e),
            "stats": {"total": 0},
            "tasks": [],
        }
