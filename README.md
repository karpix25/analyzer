## Async Video Cropper

FastAPI сервис вырезает из видео верхнюю текстовую часть и возвращает ссылки на результат. Очередь и статусы хранятся в Redis, чтобы параллельно обрабатывать много запросов.

### Быстрый старт

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
redis-server --daemonize yes  # или свой способ запуска Redis
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Конфигурация (.env)

- `PUBLIC_BASE_URL` — базовый URL, под которым раздаются файлы (`http://localhost:8000` в локале или ваш публичный домен).
- `REDIS_URL` — адрес Redis (по умолчанию `redis://localhost:6379/0`).
- `REDIS_QUEUE` — имя очереди задач (по умолчанию `video_tasks`).
- `REDIS_TASK_SET_KEY` — ключ множества с id задач (по умолчанию `video_tasks:ids`).
- `REDIS_TASK_KEY_PREFIX` — префикс ключей задач (по умолчанию `video_task:`).
- `WORKER_CONCURRENCY` — количество воркеров внутри процесса (по умолчанию `3`).

Создайте файл `.env` и укажите нужные значения, если отличаетесь от дефолтов.

### API

- `POST /detect/video` — multipart с файлом `video`, опционально `webhook_url`, `callback_data`. Возвращает `task_id` и ссылки для статуса.
- `POST /detect/video/url` — JSON `{ "video_url": "...", "webhook_url": "...", "callback_data": {...} }`.
- `GET /task/{task_id}` — статус задачи и ссылки на файлы.
- `GET /result/{task_id}/{filename}` — скачать файлы (`video_crop.mp4`, `clean_crop.mp4`, `text_crop.mp4`, `frame.jpg`, `text_frame.jpg`, `debug.jpg`, `density_profile.jpg`).

### Как это работает

1. Запрос создаёт запись задачи в Redis и кладёт job в очередь.
2. Фоновые воркеры берут задачи из Redis (`BRPOP`), качают/читают видео, определяют границу текста (EasyOCR + OpenCV), режут контент/текст через ffmpeg.
3. Статус и ссылки пишутся обратно в Redis; по `webhook_url` отправляется уведомление.

### Docker

```bash
docker build -t video-cropper .
docker run --rm -p 8000:8000 --env-file .env --link your-redis:redis video-cropper
```
