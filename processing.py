import subprocess
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import easyocr

from config import RESULT_DIR

# Инициализируем логгер
logger = logging.getLogger("app.processing")

# Initialize EasyOCR with model storage directory
logger.info("[INIT] Loading EasyOCR model...")
READER = easyocr.Reader(
    ['en', 'ru'],
    gpu=False,
    model_storage_directory=os.getenv('EASYOCR_MODULE_PATH', '/app/.EasyOCR'),
    download_enabled=False  # Don't download, use pre-downloaded models
)
logger.info("[INIT] EasyOCR model loaded")


# ---- Frame sampling ---------------------------------------------------------
def sample_frames(video_path: str, max_frames: int = 45) -> List[np.ndarray]:
    """
    Выбирает кадры из видео группами (начало, середина, конец) для лучшего анализа статики/динамики.
    Всего ~45 кадров (3 группы по 15).
    """
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    if not cap.isOpened():
        return frames

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Если видео очень короткое (< 2 сек или < 50 кадров), берем равномерно
    if total < 50:
        indices = np.linspace(0, total - 1, num=min(total, 20), dtype=np.int32)
        for idx in np.unique(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        # Стратегия 3 групп
        group_size = 15
        
        # Точки старта групп (10%, 50%, 90%)
        start_indices = [
            int(total * 0.1),
            int(total * 0.5),
            int(total * 0.9)
        ]
        
        # Корректируем, чтобы не вылезти за пределы
        start_indices = [min(s, total - group_size) for s in start_indices]
        # Корректируем, чтобы не были отрицательными
        start_indices = [max(0, s) for s in start_indices]
        
        for start_idx in start_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            for _ in range(group_size):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)

    cap.release()
    return frames


# ---- Motion-based video window detection -----------------------------------
def detect_video_window_by_motion(
    frames: List[np.ndarray],
    variance_percentile: float = 75.0,
    min_area_ratio: float = 0.10,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect video window by analyzing pixel motion across frames.
    
    Static elements (text, logos, borders) have low temporal variance.
    Dynamic elements (video content) have high temporal variance.
    
    Args:
        frames: List of sampled frames from video
        variance_percentile: Percentile for variance threshold (default: 75)
        min_area_ratio: Minimum video window size as ratio of frame (default: 0.10)
    
    Returns:
        (x, y, w, h) bounding box of video region, or None if detection fails
    """
    if len(frames) < 3:
        logger.warning("[MOTION] Not enough frames for motion detection")
        return None
    
    H, W = frames[0].shape[:2]
    logger.info(f"[MOTION] Analyzing {len(frames)} frames for motion detection...")
    
    # Convert frames to grayscale for variance calculation
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    
    # Calculate temporal variance per pixel
    frames_array = np.array(gray_frames)  # Shape: (n_frames, H, W)
    variance_map = np.var(frames_array, axis=0)  # Shape: (H, W)
    
    # Normalize variance to 0-255 range for visualization
    variance_normalized = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Calculate adaptive threshold based on variance distribution
    variance_flat = variance_map.ravel()
    variance_flat = variance_flat[variance_flat > 0]  # Ignore zero variance pixels
    
    if len(variance_flat) == 0:
        logger.warning("[MOTION] No motion detected in video (all pixels static)")
        return None
    
    threshold_value = np.percentile(variance_flat, variance_percentile)
    logger.info(f"[MOTION] Variance threshold (p{variance_percentile}): {threshold_value:.2f}")
    
    # Create binary motion mask
    motion_mask = (variance_map > threshold_value).astype(np.uint8) * 255
    
    # Morphological operations to clean up noise AND merge nearby motion regions
    # Using moderate kernel (15x15) to merge parts without losing subtle motion (e.g., sand)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find connected components
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("[MOTION] No motion regions found after thresholding")
        return None
    
    # Find largest connected component (likely the video window)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Validate minimum size
    area_ratio = (w * h) / (W * H)
    if area_ratio < min_area_ratio:
        logger.warning(f"[MOTION] Detected region too small: {area_ratio:.2%} < {min_area_ratio:.2%}")
        return None
    
    logger.info(f"[MOTION] Detected video window: x={x}, y={y}, w={w}, h={h} (area={area_ratio:.1%})")
    
    return (x, y, w, h)


# ---- Text detection helpers -------------------------------------------------
def _text_mask_improved(frame: np.ndarray, search_height_ratio: float = 0.65) -> np.ndarray:
    """Детекция белого текста на черном фоне."""
    H, W = frame.shape[:2]
    search_h = int(H * search_height_ratio)

    roi = frame[:search_h, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
    )

    mask = cv2.bitwise_and(bright, adaptive)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(mask)

    min_area = 0.00015 * H * W
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(mask_filtered, [cnt], -1, 255, -1)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    mask_filtered = cv2.dilate(mask_filtered, kernel_dilate, iterations=1)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[:search_h, :] = mask_filtered

    return full_mask


def _is_valid_text_region(x: int, y: int, w: int, h: int, W: int, H: int, strict: bool = True) -> bool:
    """Проверяет, является ли регион валидным текстовым блоком."""
    min_width = 0.20 * W if strict else 0.10 * W

    if w < min_width:
        return False

    aspect = w / max(h, 1)

    if h < 0.05 * H:
        if aspect < 1.5 or aspect > 100:
            return False
    elif h < 0.08 * H:
        if aspect < 2.0 or aspect > 50:
            return False
    else:
        if aspect < 1.2 or aspect > 50:
            return False

    if h < 5:
        return False

    if h > 0.5 * H:
        return False

    if y > 0.65 * H:
        return False

    return True


def _calculate_density_profile(contours_list: List[Tuple[int, int, int, int]], H: int, W: int, strip_height: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Рассчитывает вертикальный профиль плотности текста."""
    num_strips = (H + strip_height - 1) // strip_height
    density = np.zeros(num_strips, dtype=np.float32)
    count = np.zeros(num_strips, dtype=np.int32)

    for x, y, w, h in contours_list:
        start_strip = y // strip_height
        end_strip = min((y + h) // strip_height, num_strips - 1)

        area = w * h

        for strip_idx in range(start_strip, end_strip + 1):
            strip_top = strip_idx * strip_height
            strip_bottom = min((strip_idx + 1) * strip_height, H)

            overlap_top = max(y, strip_top)
            overlap_bottom = min(y + h, strip_bottom)

            if overlap_bottom > overlap_top:
                overlap_height = overlap_bottom - overlap_top
                overlap_area = (overlap_height / h) * area

                density[strip_idx] += overlap_area
                count[strip_idx] += 1

    strip_area = W * strip_height
    density = density / strip_area

    return density, count


def _find_main_text_region(density: np.ndarray, count: np.ndarray, strip_height: int, H: int) -> Tuple[Optional[int], Optional[int]]:
    """Находит основной текстовый регион по профилю плотности."""
    if len(density) == 0:
        return None, None

    threshold_density = max(0.02, np.percentile(density, 75))
    threshold_count = 2

    high_density_strips = (density >= threshold_density) & (count >= threshold_count)

    if not high_density_strips.any():
        threshold_count = 1
        high_density_strips = (density >= threshold_density) & (count >= threshold_count)

        if not high_density_strips.any():
            return None, None

    regions = []
    in_region = False
    region_start = 0

    for i in range(len(high_density_strips)):
        if high_density_strips[i] and not in_region:
            in_region = True
            region_start = i
        elif not high_density_strips[i] and in_region:
            in_region = False
            region_end = i - 1
            region_density = np.sum(density[region_start : region_end + 1])
            regions.append((region_start, region_end, region_density))

    if in_region:
        region_end = len(high_density_strips) - 1
        region_density = np.sum(density[region_start : region_end + 1])
        regions.append((region_start, region_end, region_density))

    if not regions:
        return None, None

    best_region = max(regions, key=lambda r: r[2])
    region_start_strip, region_end_strip, _ = best_region

    region_start_px = region_start_strip * strip_height
    region_end_px = min((region_end_strip + 1) * strip_height, H)

    logger.debug(f"[DENSITY] Found main text region: strips {region_start_strip}-{region_end_strip}, px {region_start_px}-{region_end_px}")

    return region_start_px, region_end_px


def get_text_bottom_from_contours(frame: np.ndarray) -> Tuple[Optional[int], List[Tuple[int, int, int, int]]]:
    """Используем OCR для проверки кандидатов."""
    H, W = frame.shape[:2]
    mask = _text_mask_improved(frame, search_height_ratio=0.65)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, []

    # 1. Собираем всех геометрических кандидатов
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue

        if _is_valid_text_region(x, y, w, h, W, H, strict=False):
            candidates.append((x, y, w, h))

    if not candidates:
        return None, []

    # 2. Проверяем кандидатов через OCR
    valid_contours = []
    max_bottom = 0

    logger.info(f"[OCR] Checking {len(candidates)} candidates...")

    for (x, y, w, h) in candidates:
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        roi = frame[y1:y2, x1:x2]
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        try:
            results = READER.readtext(img_gray, detail=1, paragraph=False)

            if results:
                text_content = " ".join([res[1] for res in results]).strip() # Extract text from detailed results
                if len(text_content) > 1:
                    logger.debug(f"[OCR] Found text at y={y}: '{text_content}'")
                    valid_contours.append((x, y, w, h))

                    contour_bottom = y + h
                    if contour_bottom > max_bottom:
                        max_bottom = contour_bottom
        except Exception as e:
            logger.error(f"[OCR] Error processing region: {e}")

    logger.info(f"[DEBUG] OCR confirmed {len(valid_contours)} text regions from {len(candidates)} candidates")

    if not valid_contours:
        return None, []

    # 3. Smart Separation (Gap Detection)
    valid_contours.sort(key=lambda c: c[1])

    max_gap = 0.10 * H

    prev_y, prev_h = valid_contours[0][1], valid_contours[0][3]
    final_bottom = prev_y + prev_h

    filtered_contours = [valid_contours[0]]

    for i in range(1, len(valid_contours)):
        curr_x, curr_y, curr_w, curr_h = valid_contours[i]

        gap = curr_y - (prev_y + prev_h)

        if gap > max_gap:
            logger.info(f"[GAP] Found large gap ({gap}px > {max_gap}px) at y={curr_y}. Stopping header detection.")
            break

        final_bottom = max(final_bottom, curr_y + curr_h)
        filtered_contours.append(valid_contours[i])

        prev_y = curr_y
        prev_h = curr_h

    return (final_bottom if final_bottom > 0 else None), filtered_contours


# ---- Crop estimation --------------------------------------------------------
def estimate_crop_box(frames: List[np.ndarray], task_id: str) -> Tuple[Tuple[int, int, int, int], int]:
    """Оценка crop box со среднего кадра (с downscale для скорости)."""
    H_orig, W_orig = frames[0].shape[:2]

    analysis_width = 640
    scale = 1.0
    if W_orig > analysis_width:
        scale = analysis_width / W_orig
        analysis_height = int(H_orig * scale)
        frames_small = [cv2.resize(f, (analysis_width, analysis_height)) for f in frames]
        logger.info(f"[PERF] Downscaled frames to {analysis_width}x{analysis_height} (scale={scale:.3f})")
    else:
        frames_small = frames

    task_result_dir = RESULT_DIR / task_id
    task_result_dir.mkdir(exist_ok=True)

    mid_frame_small = frames_small[len(frames_small) // 2]

    text_bottom_small, valid_contours_small = get_text_bottom_from_contours(mid_frame_small)

    if text_bottom_small is not None:
        text_bottom = int(text_bottom_small / scale)
        valid_contours = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for x, y, w, h in valid_contours_small]
    else:
        text_bottom = None
        valid_contours = []

    if text_bottom is None:
        logger.warning("[WARNING] No text on middle frame, checking all frames...")
        all_bottoms = []
        for frame in frames_small:
            bottom_s, _ = get_text_bottom_from_contours(frame)
            if bottom_s is not None:
                all_bottoms.append(int(bottom_s / scale))

        if not all_bottoms:
            text_bottom = int(0.05 * H_orig)
            logger.warning("[WARNING] No text detected on any frame!")
        else:
            text_bottom = int(np.median(all_bottoms))
            logger.info(f"[INFO] Using median from {len(all_bottoms)} frames: {text_bottom}")
    else:
        logger.info(f"[INFO] Text found on middle frame: bottom={text_bottom}, contours={len(valid_contours)}")
    
    # Set dimensions for motion detection
    H = H_orig
    W = W_orig
    margin = max(int(0.01 * H), 10)
    mid_frame_orig = frames[len(frames) // 2]
    
    
    # === NEW: Motion-based video window detection ===
    logger.info("[MOTION] Attempting motion-based video window detection...")
    motion_bbox = detect_video_window_by_motion(frames)
    
    if motion_bbox is not None:
        # Motion detection successful - use it for bbox
        mx, my, mw, mh = motion_bbox
        is_motion = True
        logger.info(f"[MOTION] Using motion-detected bbox: x={mx}, y={my}, w={mw}, h={mh}")
        
        # Combine with text_bottom: crop from text_bottom to motion bbox bottom
        if text_bottom is not None and text_bottom < my + mh:
            # Text is above motion region - use text_bottom as top
            final_y = text_bottom + margin
            final_h = (my + mh) - final_y
            logger.info(f"[HYBRID] Combined text_bottom ({text_bottom}) + motion bbox")
        else:
            # No text or text is below motion - use motion bbox as-is
            final_y = my
            final_h = mh
            logger.info(f"[MOTION] Using pure motion bbox (no text detected above)")
        
        # Use motion bbox for horizontal bounds
        bbox_rough = (mx, final_y, mw, max(1, final_h))
        
        # For debug visualization compatibility
        crop_top = final_y
        crop_height = final_h
        
    else:
        # Motion detection failed - fallback to original algorithm
        is_motion = False
        logger.warning("[MOTION] Motion detection failed, using fallback (crop from text_bottom)")
        
        margin = max(int(0.01 * H), 10)
        crop_top = text_bottom + margin
        
        logger.info(f"[CROP] text_bottom={text_bottom}, margin={margin}, crop_top={crop_top}")
        
        crop_height = H - crop_top
        min_crop_h = max(int(0.4 * H), 250)
        
        if crop_height < min_crop_h:
            logger.warning(f"[WARNING] Crop height {crop_height} < min {min_crop_h}")
            crop_top = H - min_crop_h
            crop_height = min_crop_h
            
            if crop_top <= text_bottom:
                logger.error(f"[ERROR] Cannot satisfy min height! Using minimal margin.")
                crop_top = text_bottom + max(int(0.01 * H), 8)
                crop_height = H - crop_top
        
        bbox_rough = (0, crop_top, W, crop_height)
    
    # Continue with existing logic for debug visualization
    x, y, w, h = bbox_rough
    debug_frame = mid_frame_orig.copy()

    mask_debug = _text_mask_improved(mid_frame_orig, search_height_ratio=0.65)
    cv2.imwrite(str(task_result_dir / "mask_debug.jpg"), mask_debug)

    contours_all, _ = cv2.findContours(mask_debug, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_with_contours = mid_frame_orig.copy()

    valid_set = set(valid_contours)

    for cnt in contours_all:
        x, y, w, h = cv2.boundingRect(cnt)
        if (x, y, w, h) in valid_set:
            cv2.rectangle(debug_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(debug_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imwrite(str(task_result_dir / "contours_debug.jpg"), debug_with_contours)

    if valid_contours:
        strip_height = max(40, H // 25)
        density, count = _calculate_density_profile(valid_contours, H, W, strip_height)

        profile_img = np.zeros((H, 300, 3), dtype=np.uint8)

        max_density = density.max() if density.max() > 0 else 1.0
        max_count = count.max() if count.max() > 0 else 1

        for i in range(len(density)):
            y_top = i * strip_height
            y_bottom = min((i + 1) * strip_height, H)

            bar_width_density = int((density[i] / max_density) * 150)
            cv2.rectangle(profile_img, (0, y_top), (bar_width_density, y_bottom), (255, 0, 0), -1)

            bar_width_count = int((count[i] / max_count) * 150)
            cv2.rectangle(profile_img, (150, y_top), (150 + bar_width_count, y_bottom), (0, 255, 0), -1)

        for i in range(0, H, strip_height):
            cv2.line(profile_img, (0, i), (300, i), (50, 50, 50), 1)

        cv2.putText(profile_img, "Density", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(profile_img, "Count", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(task_result_dir / "density_profile.jpg"), profile_img)

    cv2.line(debug_frame, (0, text_bottom), (W, text_bottom), (0, 255, 0), 4)
    cv2.line(debug_frame, (0, crop_top), (W, crop_top), (0, 0, 255), 4)

    overlay_text = debug_frame.copy()
    cv2.rectangle(overlay_text, (0, 0), (W, text_bottom), (0, 0, 255), -1)
    debug_frame = cv2.addWeighted(debug_frame, 0.85, overlay_text, 0.15, 0)

    overlay_margin = debug_frame.copy()
    cv2.rectangle(overlay_margin, (0, text_bottom), (W, crop_top), (0, 255, 255), -1)
    debug_frame = cv2.addWeighted(debug_frame, 0.90, overlay_margin, 0.10, 0)

    overlay_keep = debug_frame.copy()
    cv2.rectangle(overlay_keep, (0, crop_top), (W, H), (0, 255, 0), -1)
    debug_frame = cv2.addWeighted(debug_frame, 0.85, overlay_keep, 0.15, 0)

    info_y = 30
    line_height = 35
    cv2.rectangle(debug_frame, (5, 5), (650, info_y + line_height * 5), (0, 0, 0), -1)

    cv2.putText(debug_frame, f"text_bottom: {text_bottom} ({text_bottom / H * 100:.1f}%) GREEN", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    info_y += line_height

    cv2.putText(debug_frame, f"crop_top: {crop_top} ({crop_top / H * 100:.1f}%) RED", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    info_y += line_height

    cv2.putText(debug_frame, f"margin: {margin}px ({margin / H * 100:.1f}%)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    info_y += line_height

    cv2.putText(debug_frame, f"crop_height: {crop_height} ({crop_height / H * 100:.1f}%)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += line_height

    cv2.putText(debug_frame, f"valid_contours: {len(valid_contours)} (density filtered)", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    legend_y = H - 120
    cv2.rectangle(debug_frame, (10, legend_y), (550, H - 10), (0, 0, 0), -1)
    cv2.putText(debug_frame, "RED area = TEXT (will be removed)", (20, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(debug_frame, "YELLOW area = MARGIN 1% (safety gap)", (20, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(debug_frame, "GREEN area = CONTENT (will be kept)", (20, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(str(task_result_dir / "debug.jpg"), debug_frame)

    return bbox_rough, text_bottom, is_motion


# ---- Frame selection & refinement ------------------------------------------
def select_best_frame(frames: List[np.ndarray], bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
    """Выбор самого резкого кадра."""
    x, y, w, h = bbox
    best_frame = frames[0]
    best_score = -1.0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        lap = cv2.Laplacian(crop, cv2.CV_32F, ksize=3)
        score = float(np.abs(lap).mean())
        if score > best_score:
            best_score = score
            best_frame = frame

    return best_frame, best_score


# ---- Video format detection ------------------------------------------------

def _has_rounded_corners(frame: np.ndarray) -> bool:
    """Проверяет наличие скругленных углов (черные области в углах)."""
    h, w = frame.shape[:2]
    
    # Проверяем углы (размер зависит от размера кадра)
    corner_size = min(50, h // 10, w // 10)
    
    if corner_size < 10:
        return False
    
    # Проверяем 4 угла
    corners = [
        frame[:corner_size, :corner_size],  # Top-left
        frame[:corner_size, -corner_size:],  # Top-right
        frame[-corner_size:, :corner_size],  # Bottom-left
        frame[-corner_size:, -corner_size:],  # Bottom-right
    ]
    
    dark_corners = 0
    for corner in corners:
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        median_val = np.median(gray)
        # Если угол темный (< 30) - это черная область
        if median_val < 30:
            dark_corners += 1
    
    # Если 3+ угла темные - скругленные углы
    return dark_corners >= 3


def _has_bottom_overlay(frame: np.ndarray) -> bool:
    """Проверяет наличие текста/иконок в нижней части."""
    h, w = frame.shape[:2]
    
    # Анализируем нижние 30%
    bottom_region = frame[int(h * 0.70):, :]
    
    if bottom_region.shape[0] < 20:
        return False
    
    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    
    # Ищем края (текст/иконки создают края)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Если много краёв (> 2%) - вероятно есть текст/иконки
    if edge_density > 0.02:
        return True
    
    # Проверяем неоднородность
    row_stds = gray.std(axis=1)
    uniform_rows = np.sum(row_stds < 5.0) / len(row_stds)
    
    # Если НЕ однородный (< 80% однородных строк) - возможно оверлей
    return uniform_rows < 0.80


def detect_video_format(frame: np.ndarray) -> str:
    """
    Определяет формат видео.
    
    Returns:
        'rounded_corners' - видео со скругленными углами
        'with_overlay' - видео с оверлеем снизу (текст/иконки)
        'standard' - стандартное видео
    """
    # Проверяем в порядке специфичности
    if _has_rounded_corners(frame):
        return 'rounded_corners'
    
    if _has_bottom_overlay(frame):
        return 'with_overlay'
    
    return 'standard'


def refine_crop_rect(
    frame: np.ndarray, 
    x: int, 
    y: int, 
    w: int, 
    h: int,
    task_id: Optional[str] = None,
    save_debug: bool = True,
    full_frame: Optional[np.ndarray] = None,
    roi_offset_y: int = 0,
    video_format: str = 'standard'
) -> Tuple[int, int, int, int]:
    """Уточняет область обрезки, убирая черные полосы (letterbox) внутри указанного ROI.
    
    Args:
        frame: Исходный кадр (может быть ROI)
        x, y, w, h: Входной bbox (rough crop)
        task_id: ID задачи для сохранения debug изображения
        save_debug: Сохранять ли debug визуализацию
        full_frame: Полный кадр для debug (если frame это ROI)
        roi_offset_y: Смещение Y ROI относительно полного кадра (например, text_bottom)
        video_format: Формат видео ('standard', 'rounded_corners', 'with_overlay')
    
    Returns:
        Refined bbox (x, y, w, h) - координаты внутри frame
    """
    if w <= 0 or h <= 0:
        return x, y, w, h

    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return x, y, w, h
    
    # Логируем определённый формат
    if task_id:
        logger.info(f"[TASK {task_id}] Video format: {video_format}")
    
    # АДАПТИВНАЯ ОБРАБОТКА: Применяем специфичную стратегию для формата
    if video_format == 'rounded_corners':
        # Для скругленных углов: более агрессивная обрезка черных областей
        logger.info(f"[FORMAT] Applying rounded_corners strategy")
        # Используем более низкий порог для черного цвета
        black_threshold = 20  # вместо стандартного 40
    elif video_format == 'with_overlay':
        # Для видео с оверлеем: фокус на обрезке снизу
        logger.info(f"[FORMAT] Applying with_overlay strategy")
        black_threshold = 40  # стандартный порог
    else:
        # Стандартная обработка
        black_threshold = 40


    def _background_aware_mask(bgr: np.ndarray) -> np.ndarray:
        """Строит маску контента, убирая однотонные поля любого цвета (черный/белый/оранжевый и т.п.)."""
        h_roi, w_roi = bgr.shape[:2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        # Уменьшаем edge для лучшей детекции тонких полос
        edge = max(1, int(0.04 * min(h_roi, w_roi)))
        
        # Используем все 4 стороны для определения фона
        # Если все стороны одного цвета - это фон
        edges = [
            lab[:edge, :, :],          # Верх
            lab[:, :edge, :],          # Левая сторона
            lab[:, w_roi - edge :, :], # Правая сторона
            lab[h_roi - edge:, :, :],  # Низ (добавлено для детекции бордового фона)
        ]
        edges_stack = np.concatenate([e.reshape(-1, 3) for e in edges], axis=0)
        bg_color = np.median(edges_stack, axis=0)

        dist = np.linalg.norm(lab.astype(np.float32) - bg_color.astype(np.float32), axis=2)
        
        # Для порога используем все 4 стороны
        dist_edges = np.concatenate(
            [
                dist[:edge, :].ravel(),
                dist[:, :edge].ravel(),
                dist[:, w_roi - edge :].ravel(),
                dist[h_roi - edge:, :].ravel(),  # Низ (добавлено)
            ]
        )
        edge_threshold = float(np.percentile(dist_edges, 95)) if dist_edges.size else 0.0
        # Относительный порог: 2% от среднего размера видео
        # Для 360px → ~8.6, для 720px → ~17.3
        # Адаптируется к размеру видео автоматически
        base_threshold = 0.02 * np.mean([h_roi, w_roi])
        threshold = max(base_threshold, edge_threshold * 1.0)

        mask = (dist > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Убран MORPH_OPEN - он удалял тонкие края видео
        # Оставлен только MORPH_CLOSE для заполнения дыр внутри контента
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    # КРИТИЧНО: Проверяем нижний край ДО расчета bounding rect
    # Это позволяет обнаружить большие цветные полосы снизу
    
    def _find_video_bottom_edge(bgr_roi: np.ndarray) -> Optional[int]:
        """Находит нижнюю границу видео по резкому горизонтальному переходу."""
        h, w = bgr_roi.shape[:2]
        
        if h < 40:
            return None
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        
        # Детектируем горизонтальные края (переходы) с помощью Sobel
        # dy=1 означает вертикальный градиент (горизонтальные края)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobelx_abs = np.abs(sobelx)
        
        # Считаем среднюю силу горизонтального края для каждой строки
        edge_strength = sobelx_abs.mean(axis=1)
        
        # Ищем сильные горизонтальные края в нижней половине
        # (граница видео обычно в нижней части)
        bottom_half_start = h // 2
        bottom_edges = edge_strength[bottom_half_start:]
        
        if len(bottom_edges) == 0:
            return None
        
        # Находим самые сильные края (более строгий порог)
        threshold = np.percentile(bottom_edges, 95)  # Увеличено с 85 до 95
        strong_edges_indices = np.where(bottom_edges > threshold)[0]
        
        if len(strong_edges_indices) == 0:
            return None
        
        # Берём первый сильный край (самый верхний в нижней половине)
        # Это вероятно граница между видео и оверлеем
        edge_idx = strong_edges_indices[0] + bottom_half_start
        
        # Проверяем что граница в разумном месте (не слишком высоко)
        if edge_idx < h * 0.3:  # Не выше 30% от высоты
            return None
        
        return edge_idx
    
    def _is_uniform_region(bgr_region: np.ndarray) -> bool:
        """Проверяет что регион однородный (фон)."""
        if bgr_region.shape[0] < 10:
            return False
        
        gray = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2GRAY)
        
        # Проверяем однородность по строкам
        row_stds = gray.std(axis=1)
        
        # Если большинство строк однородные (std < 5)
        uniform_rows = np.sum(row_stds < 5.0) / len(row_stds)
        
        return uniform_rows > 0.70  # 70%+ строк однородные
    
    
    def _detect_bottom_text_overlay(bgr_roi: np.ndarray) -> int:
        """Определяет наличие текста/иконок в нижней части и возвращает высоту для обрезки.
        
        Различает:
        - Текст как часть видео (на сложном фоне) → НЕ обрезаем
        - Текст как overlay (на однородном фоне) → обрезаем
        """
        h, w = bgr_roi.shape[:2]
        if h < 50:
            return 0
        
        # Анализируем нижние 40% кадра (увеличено с 30% для лучшего покрытия текста)
        bottom_height = int(h * 0.4)
        bottom_region = bgr_roi[h - bottom_height:, :]
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Используем адаптивную бинаризацию для лучшей детекции текста на темном фоне
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Детекция краёв на бинарном изображении
        edges = cv2.Canny(binary, 50, 150)
        
        # Находим контуры
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтруем мелкие контуры (текст = много мелких контуров)
        small_contours = [c for c in contours if 10 < cv2.contourArea(c) < 500]
        
        # Плотность мелких контуров
        contour_density = len(small_contours) / (bottom_region.shape[0] * bottom_region.shape[1] / 1000)
        
        # Если плотность низкая - текста нет
        if contour_density <= 0.2:
            return 0
        
        # НОВАЯ ЛОГИКА: Проверяем фон под текстом
        # Если фон однородный (std < 15) → это overlay на черной полосе → обрезаем
        # Если фон сложный (std > 15) → это часть видео → НЕ обрезаем
        
        background_std = gray.std()
        
        if background_std < 15:
            # Однородный фон → overlay (watermark на черной полосе)
            logger.info(f"[TEXT] Detected OVERLAY text (density={contour_density:.2f}, bg_std={background_std:.1f}), trimming {bottom_height}px")
            return bottom_height
        else:
            # Сложный фон → часть видео (текст на телефоне, в сцене)
            logger.info(f"[TEXT] Detected VIDEO text (density={contour_density:.2f}, bg_std={background_std:.1f}), NOT trimming")
            return 0
    
    
    def _detect_bottom_uniform_strip(bgr_roi: np.ndarray) -> int:
        """Определяет, сколько пикселей снизу нужно обрезать из-за однотонной полосы."""
        h_check, w_check = bgr_roi.shape[:2]
        if h_check < 20:
            return 0
        
        # НОВЫЙ ПОДХОД: Сначала пробуем найти резкую границу через edge detection
        # Это помогает когда видео темное и сливается с темным фоном
        edge_boundary = _find_video_bottom_edge(bgr_roi)
        
        if edge_boundary is not None:
            # Нашли резкую границу - проверяем что ниже действительно фон
            below_height = h_check - edge_boundary
            
            # Более строгий диапазон: 10-40% (было 5-60%)
            if below_height > h_check * 0.10 and below_height < h_check * 0.40:
                # Проверяем что регион ниже действительно однородный (фон)
                below_region = bgr_roi[edge_boundary:, :]
                
                if _is_uniform_region(below_region):
                    logger.info(f"[EDGE] Validated boundary at y={edge_boundary}, trimming {below_height}px")
                    return below_height
                else:
                    logger.info(f"[EDGE] Rejected boundary at y={edge_boundary} - region below is not uniform")
            else:
                logger.info(f"[EDGE] Rejected boundary at y={edge_boundary} - out of range (10-40%)")
        
        
        # FALLBACK: Используем построчный анализ (существующая логика)
        # Анализируем нижние 30% кадра (увеличено для лучшего покрытия)
        bottom_zone_height = max(20, int(h_check * 0.30))
        bottom_zone = bgr_roi[h_check - bottom_zone_height:, :]
        
        # Конвертируем в LAB для лучшего анализа цвета
        lab_bottom = cv2.cvtColor(bottom_zone, cv2.COLOR_BGR2LAB)
        gray_bottom = cv2.cvtColor(bottom_zone, cv2.COLOR_BGR2GRAY)
        
        # Анализируем каждую строку снизу вверх
        rows_to_trim = 0
        max_trim = int(h_check * 0.50)  # Максимум 50% можно обрезать
        
        for i in range(bottom_zone_height - 1, -1, -1):
            row_lab = lab_bottom[i, :, :]
            row_gray = gray_bottom[i, :]
            
            # НОВЫЙ ПОДХОД: Анализируем пиксели, а не всю строку
            # Определяем доминирующий цвет строки
            median_lab = np.median(row_lab, axis=0)
            median_gray = np.median(row_gray)
            
            # Считаем сколько пикселей близки к доминирующему цвету
            dist_from_median_lab = np.linalg.norm(row_lab.astype(np.float32) - median_lab.astype(np.float32), axis=1)
            dist_from_median_gray = np.abs(row_gray.astype(np.float32) - median_gray)
            
            # Пиксель считается "однотонным" если он близок к медиане
            uniform_pixels_lab = dist_from_median_lab < 15.0  # Порог в LAB пространстве
            uniform_pixels_gray = dist_from_median_gray < 10.0  # Порог в grayscale
            
            # Комбинируем: пиксель однотонный если оба условия выполнены
            uniform_pixels = uniform_pixels_lab & uniform_pixels_gray
            uniform_ratio = np.sum(uniform_pixels) / len(uniform_pixels)
            
            # Строка считается фоном если:
            # 1. Высокий uniform_ratio (75%+) - однородная строка
            # 2. ИЛИ низкая контрастность (полупрозрачный watermark)
            
            if uniform_ratio >= 0.75:
                # Однородная строка - проверяем цвет
                is_background = (
                    median_gray < 40 or median_gray > 210 or 
                    (median_gray > 60 and median_gray < 200)
                )
            else:
                # Неоднородная строка (есть текст?) - проверяем контрастность
                contrast = np.std(row_gray)
                
                # Если контрастность низкая (<15) - это полупрозрачный watermark
                # Игнорируем его и считаем фоном
                is_low_contrast_overlay = contrast < 15
                
                if is_low_contrast_overlay and (median_gray < 50 or median_gray > 200):
                    # Полупрозрачный текст на темном/светлом фоне → фон
                    is_background = True
                else:
                    # Высокая контрастность → реальный контент
                    is_background = False
            
            if is_background:
                rows_to_trim += 1
                if rows_to_trim >= max_trim:
                    break
            else:
                # Если нашли неоднородную строку, останавливаемся
                # Даём запас только если уже нашли достаточно
                if rows_to_trim > 10:
                    rows_to_trim -= 5  # Больше запаса для больших полос
                elif rows_to_trim > 5:
                    rows_to_trim -= 2
                elif rows_to_trim > 0:
                    rows_to_trim = 0
                break
        
        return rows_to_trim
    
    # Применяем проверку нижнего края на ИСХОДНОМ ROI
    # ВАЖНО: НЕ изменяем roi и h, только запоминаем сколько нужно обрезать!
    
    # Сначала проверяем наличие текста/иконок внизу
    text_trim_pixels = _detect_bottom_text_overlay(roi)
    
    # Затем проверяем однородную полосу
    bottom_trim_pixels = _detect_bottom_uniform_strip(roi)
    
    # Используем максимум из двух детекций
    bottom_trim_pixels = max(text_trim_pixels, bottom_trim_pixels)
    
    if bottom_trim_pixels > 0:
        logger.info(f"[BOTTOM_STRIP] Detected bottom overlay: {bottom_trim_pixels}px ({bottom_trim_pixels/h*100:.1f}%)")
    
    # Применяем background-aware mask на ИСХОДНОМ ROI
    # (не обрезаем, чтобы сохранить координаты)
    color_mask = _background_aware_mask(roi)

    if cv2.countNonZero(color_mask) < 10:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, color_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Если нашли bottom trim, применяем его
        if bottom_trim_pixels > 0:
            return x, y, w, max(1, h - bottom_trim_pixels)
        return x, y, w, h

    all_points = np.concatenate(contours)
    rx, ry, rw, rh = cv2.boundingRect(all_points)

    margin = 2
    rx = max(0, rx - margin)
    ry = max(0, ry - margin)
    rw = min(w - rx, rw + 2 * margin)
    rh = min(h - ry, rh + 2 * margin)

    # Дополнительное обрезание однотонных полос по краям с индивидуальными ограничениями.
    def _trim_uniform_edges(gray_roi: np.ndarray, max_ratio_std: float = 0.18, max_ratio_bottom: float = 0.40) -> Tuple[int, int, int, int]:
        """Возвращает (left, top, width, height) после среза однотонных полос."""
        r_h, r_w = gray_roi.shape
        row_mean = gray_roi.mean(axis=1)
        row_std = gray_roi.std(axis=1)
        col_mean = gray_roi.mean(axis=0)
        col_std = gray_roi.std(axis=0)

        # Пороги однородности - увеличены для более строгой детекции
        # Увеличено с 5.0 до 8.0 чтобы не обрезать края видео с небольшой текстурой
        std_thresh_row = 8.0  # Только очень однородные области
        std_thresh_col = 8.0  # Только очень однородные области

        max_trim_rows_std = int(r_h * max_ratio_std)
        max_trim_cols_std = int(r_w * max_ratio_std)
        max_trim_rows_bottom = int(r_h * max_ratio_bottom)

        def _scan_forward(arr_mean, arr_std, std_thresh, limit):
            if limit <= 0 or len(arr_mean) == 0:
                return 0
            
            # Определяем эталонный цвет фона по краю
            bg_ref = arr_mean[0]
            
            # ADAPTIVE tolerance: для темных фонов (черный) нужен больший допуск,
            # так как темное видео (15-20) близко к черному фону (5), но это разные объекты
            if bg_ref < 30:  # Темный фон (черный)
                color_tolerance = 20.0
            elif bg_ref > 225:  # Светлый фон (белый)
                color_tolerance = 20.0
            else:  # Средние тона
                color_tolerance = 12.0
            
            idx = 0
            while idx < limit:
                current_mean = arr_mean[idx]
                current_std = arr_std[idx]
                
                # Проверяем только однородность и непрерывность цвета
                # Убрана проверка mean < 40 - она вызывала over-trimming
                
                # 1. Проверка на однородность (нет текстуры/шума)
                if current_std > std_thresh:
                    break
                
                # 2. Проверка на непрерывность цвета (защита от перехода Фон -> Объект)
                if abs(current_mean - bg_ref) > color_tolerance:
                    break
                    
                idx += 1
            return idx

        def _scan_backward(arr_mean, arr_std, std_thresh, limit):
            if limit <= 0 or len(arr_mean) == 0:
                return len(arr_mean)
            
            # Start from the END (inclusive length)
            idx = len(arr_mean)
            limit_idx = len(arr_mean) - limit
            
            # Reference color at the very bottom/right edge
            bg_ref = arr_mean[len(arr_mean) - 1]
            
            # ADAPTIVE tolerance (same logic as forward)
            if bg_ref < 30:  # Темный фон (черный)
                color_tolerance = 20.0
            elif bg_ref > 225:  # Светлый фон (белый)
                color_tolerance = 20.0
            else:  # Средние тона
                color_tolerance = 12.0

            # Move upwards/leftwards
            while idx > limit_idx and idx > 0:
                curr_check_idx = idx - 1
                current_mean = arr_mean[curr_check_idx]
                current_std = arr_std[curr_check_idx]
                
                # Проверяем только однородность и непрерывность цвета
                # Убрана проверка mean < 40 - симметрично с _scan_forward
                
                # Проверка на однородность
                if current_std > std_thresh:
                    break
                
                # Проверка на непрерывность цвета
                if abs(current_mean - bg_ref) > color_tolerance:
                    break
                    
                idx -= 1
            return idx

        top = _scan_forward(row_mean, row_std, std_thresh_row, max_trim_rows_std)
        bottom = _scan_backward(row_mean, row_std, std_thresh_row, max_trim_rows_bottom)
        left = _scan_forward(col_mean, col_std, std_thresh_col, max_trim_cols_std)
        right = _scan_backward(col_mean, col_std, std_thresh_col, max_trim_cols_std)

        new_w = max(1, right - left)
        new_h = max(1, bottom - top)
        return left, top, new_w, new_h

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    trim_l, trim_t, trim_w, trim_h = _trim_uniform_edges(gray_roi)

    rx = rx + trim_l
    ry = ry + trim_t
    rw = min(rw - trim_l, trim_w)
    rh = min(rh - trim_t, trim_h)

    # Боковые границы: срез не более 10%, низ может срезаться до 40%
    max_side_crop_x = int(0.10 * w)
    max_top_crop = int(0.10 * h)
    max_bottom_crop = int(0.40 * h)

    # Ограничиваем слева/справа
    if rx > max_side_crop_x:
        overshoot = rx - max_side_crop_x
        rx = max_side_crop_x
        rw = max(1, rw + overshoot)
    right_cut = w - (rx + rw)
    if right_cut > max_side_crop_x:
        excess = right_cut - max_side_crop_x
        rw = max(1, rw + excess)

    # Ограничиваем сверху
    if ry > max_top_crop:
        overshoot = ry - max_top_crop
        ry = max_top_crop
        rh = max(1, rh + overshoot)
    # Ограничиваем снизу (но даём больше свободы)
    bottom_cut = h - (ry + rh)
    if bottom_cut > max_bottom_crop:
        excess = bottom_cut - max_bottom_crop
        rh = max(1, rh + excess)

    # Padding убран - это костыль который маскирует проблемы детекции
    # Если границы определены неправильно, нужно улучшать алгоритм, а не добавлять запас
    # Доверяем _background_aware_mask и _trim_uniform_edges
    
    # Оставляем координаты как есть, без расширения

    final_x = x + rx
    final_y = y + ry
    final_w = rw
    final_h = rh
    
    # Учитываем обнаруженную нижнюю полосу
    if bottom_trim_pixels > 0:
        # Обрезаем снизу: уменьшаем высоту на величину детектированной полосы
        available_height_from_bottom = h - (ry + rh)  # Сколько уже обрезано снизу
        additional_bottom_trim = max(0, bottom_trim_pixels - available_height_from_bottom)
        if additional_bottom_trim > 0:
            final_h = max(1, final_h - additional_bottom_trim)
            logger.info(f"[AUTOCROP] Applied additional bottom trim: {additional_bottom_trim}px")

    logger.info(f"[AUTOCROP] Original: {w}x{h}, Refined: {final_w}x{final_h} (Removed top={ry}, bottom={h - (ry + final_h)}, left={rx}, right={w - (rx + rw)})")

    # === DEBUG VISUALIZATION ===
    if save_debug and task_id:
        try:
            task_result_dir = RESULT_DIR / task_id
            task_result_dir.mkdir(exist_ok=True)
            
            # Используем полный кадр для debug если передан, иначе текущий frame
            if full_frame is not None:
                debug_img = full_frame.copy()
                H_full, W_full = full_frame.shape[:2]
            else:
                debug_img = frame.copy()
                H_full, W_full = frame.shape[:2]
            
            # Пересчитываем координаты в глобальные (полного кадра)
            global_x = x
            global_y = y + roi_offset_y
            global_final_x = final_x
            global_final_y = final_y + roi_offset_y
            
            # 1. Рисуем входной bbox (синий) - rough crop
            cv2.rectangle(debug_img, (global_x, global_y), (global_x + w, global_y + h), (255, 0, 0), 3)
            cv2.putText(debug_img, "INPUT (rough)", 
                       (global_x + 10, global_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # 2. Рисуем выходной bbox (зелёный) - refined/clean
            cv2.rectangle(debug_img, (global_final_x, global_final_y), 
                         (global_final_x + final_w, global_final_y + final_h), 
                         (0, 255, 0), 3)
            cv2.putText(debug_img, "OUTPUT (clean)", 
                       (global_final_x + 10, global_final_y + final_h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 3. Показываем обрезанные зоны полупрозрачным красным
            overlay = debug_img.copy()
            
            # Верх (обрезано)
            if ry > 0:
                cv2.rectangle(overlay, (global_x, global_y), (global_x + w, global_y + ry), (0, 0, 255), -1)
            
            # Низ (обрезано)
            if h - (ry + rh) > 0:
                cv2.rectangle(overlay, (global_x, global_y + ry + rh), (global_x + w, global_y + h), (0, 0, 255), -1)
            
            # Лево (обрезано)
            if rx > 0:
                cv2.rectangle(overlay, (global_x, global_y), (global_x + rx, global_y + h), (0, 0, 255), -1)
            
            # Право (обрезано)
            if w - (rx + rw) > 0:
                cv2.rectangle(overlay, (global_x + rx + rw, global_y), (global_x + w, global_y + h), (0, 0, 255), -1)
            
            # Смешиваем overlay с оригиналом (transparency)
            cv2.addWeighted(overlay, 0.3, debug_img, 0.7, 0, debug_img)
            
            # 4. Добавляем информацию о обрезке
            info_y = 40
            line_h = 35
            # Чёрный фон для текста
            cv2.rectangle(debug_img, (5, 5), (600, info_y + line_h * 7), (0, 0, 0), -1)
            
            cv2.putText(debug_img, "CLEAN CROP REFINEMENT:", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += line_h
            
            cv2.putText(debug_img, f"Input:  {w}x{h} at ({global_x}, {global_y})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            info_y += line_h
            
            cv2.putText(debug_img, f"Output: {final_w}x{final_h} at ({global_final_x}, {global_final_y})", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += line_h
            
            cv2.putText(debug_img, f"Trimmed: T={ry}, B={h - (ry + rh)}, L={rx}, R={w - (rx + rw)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            info_y += line_h
            
            if roi_offset_y > 0:
                cv2.putText(debug_img, f"ROI offset Y: {roi_offset_y}px (from text_bottom)", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                info_y += line_h
            
            if bottom_trim_pixels > 0:
                cv2.putText(debug_img, f"Bottom strip detected: {bottom_trim_pixels}px", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Сохраняем
            debug_path = task_result_dir / "clean_crop_debug.jpg"
            cv2.imwrite(str(debug_path), debug_img)
            logger.info(f"[DEBUG] Saved clean_crop debug: {debug_path}")
            
        except Exception as e:
            logger.error(f"[DEBUG] Failed to save clean_crop debug image: {e}")

    return final_x, final_y, final_w, final_h


# ---- Video crops -----------------------------------------------------------
def crop_video_ffmpeg(src: Path, task_id: str, text_bottom: int, content_bbox: Tuple[int, int, int, int], clean_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Обрезка видео через ffmpeg.
    Возвращает пути: (content_video, text_video, clean_video)
    """
    task_result_dir = RESULT_DIR / task_id
    task_result_dir.mkdir(exist_ok=True)

    dst_content = task_result_dir / "video_crop.mp4"
    dst_text = task_result_dir / "text_crop.mp4"
    dst_clean = task_result_dir / "clean_crop.mp4"

    for dst in [dst_content, dst_text, dst_clean]:
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass

    x, y, w, h = content_bbox
    cmd_content = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-filter:v",
        f"crop={w}:{h}:{x}:{y}",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "veryfast",
        "-c:a",
        "copy",
        str(dst_content),
    ]
    subprocess.run(cmd_content, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if clean_bbox:
        cx, cy, cw, ch = clean_bbox
        if (cx, cy, cw, ch) != (x, y, w, h):
            cmd_clean = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-filter:v",
                f"crop={cw}:{ch}:{cx}:{cy}",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-preset",
                "veryfast",
                "-c:a",
                "copy",
                str(dst_clean),
            ]
            subprocess.run(cmd_clean, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            import shutil

            shutil.copy(dst_content, dst_clean)
    else:
        dst_clean = None

    if text_bottom > 50:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width",
            "-of",
            "csv=s=x:p=0",
            str(src),
        ]
        try:
            video_width = int(subprocess.check_output(probe_cmd).strip())

            cmd_text = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-filter:v",
                f"crop={video_width}:{text_bottom}:0:0",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-preset",
                "veryfast",
                "-c:a",
                "copy",
                str(dst_text),
            ]
            subprocess.run(cmd_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            dst_text = None
    else:
        dst_text = None

    return dst_content, dst_text, dst_clean
