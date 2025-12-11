import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import easyocr

from config import RESULT_DIR

# Инициализируем EasyOCR один раз
print("[INIT] Loading EasyOCR model...")
reader = easyocr.Reader(["en", "ru"], gpu=False)
print("[INIT] EasyOCR model loaded.")


# ---- Frame sampling ---------------------------------------------------------
def sample_frames(video_path: str, max_frames: int = 10) -> List[np.ndarray]:
    """Равномерно выбирает кадры из видео."""
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    if not cap.isOpened():
        return frames

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(round(fps * 3.0)) if fps and fps > 1e-3 else 90

    if total > 0:
        if start_frame >= total:
            start_frame = 0
        available = max(total - start_frame, 1)
        count = min(max_frames, available)
        indices = np.linspace(start_frame, total - 1, num=count, dtype=np.int32)
        for idx in np.unique(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        skipped = 0
        while skipped < start_frame:
            ok, _ = cap.read()
            if not ok:
                break
            skipped += 1
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)

    cap.release()
    return frames


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

    print(f"[DENSITY] Found main text region: strips {region_start_strip}-{region_end_strip}, px {region_start_px}-{region_end_px}")

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

    print(f"[OCR] Checking {len(candidates)} candidates...")

    for (x, y, w, h) in candidates:
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        roi = frame[y1:y2, x1:x2]

        try:
            results = reader.readtext(roi, detail=0, paragraph=True)

            if results:
                text_content = " ".join(results).strip()
                if len(text_content) > 1:
                    print(f"[OCR] Found text at y={y}: '{text_content}'")
                    valid_contours.append((x, y, w, h))

                    contour_bottom = y + h
                    if contour_bottom > max_bottom:
                        max_bottom = contour_bottom
        except Exception as e:
            print(f"[OCR] Error processing region: {e}")

    print(f"[DEBUG] OCR confirmed {len(valid_contours)} text regions from {len(candidates)} candidates")

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
            print(f"[GAP] Found large gap ({gap}px > {max_gap}px) at y={curr_y}. Stopping header detection.")
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
        print(f"[PERF] Downscaled frames to {analysis_width}x{analysis_height} (scale={scale:.3f})")
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
        print("[WARNING] No text on middle frame, checking all frames...")
        all_bottoms = []
        for frame in frames_small:
            bottom_s, _ = get_text_bottom_from_contours(frame)
            if bottom_s is not None:
                all_bottoms.append(int(bottom_s / scale))

        if not all_bottoms:
            text_bottom = int(0.05 * H_orig)
            print("[WARNING] No text detected on any frame!")
        else:
            text_bottom = int(np.median(all_bottoms))
            print(f"[INFO] Using median from {len(all_bottoms)} frames: {text_bottom}")
    else:
        print(f"[INFO] Text found on middle frame: bottom={text_bottom}, contours={len(valid_contours)}")

    H = H_orig
    W = W_orig

    margin = max(int(0.01 * H), 10)
    crop_top = text_bottom + margin

    print(f"[CROP] text_bottom={text_bottom}, margin={margin}, crop_top={crop_top}")

    crop_height = H - crop_top
    min_crop_h = max(int(0.4 * H), 250)

    if crop_height < min_crop_h:
        print(f"[WARNING] Crop height {crop_height} < min {min_crop_h}")
        crop_top = H - min_crop_h
        crop_height = min_crop_h

        if crop_top <= text_bottom:
            print(f"[ERROR] Cannot satisfy min height! Using minimal margin.")
            crop_top = text_bottom + max(int(0.01 * H), 8)
            crop_height = H - crop_top

    mid_frame_orig = frames[len(frames) // 2]
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

    return (0, int(crop_top), W, int(crop_height)), text_bottom


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


def refine_crop_rect(frame: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """Уточняет область обрезки, убирая черные полосы (letterbox) внутри указанного ROI."""
    if w <= 0 or h <= 0:
        return x, y, w, h

    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return x, y, w, h

    def _background_aware_mask(bgr: np.ndarray) -> np.ndarray:
        """Строит маску контента, убирая однотонные поля любого цвета (черный/белый/оранжевый и т.п.)."""
        h_roi, w_roi = bgr.shape[:2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        edge = max(1, int(0.06 * min(h_roi, w_roi)))
        edges = [
            lab[:edge, :, :],
            lab[h_roi - edge :, :, :],
            lab[:, :edge, :],
            lab[:, w_roi - edge :, :],
        ]
        edges_stack = np.concatenate([e.reshape(-1, 3) for e in edges], axis=0)
        bg_color = np.median(edges_stack, axis=0)

        dist = np.linalg.norm(lab.astype(np.float32) - bg_color.astype(np.float32), axis=2)
        dist_edges = np.concatenate(
            [
                dist[:edge, :].ravel(),
                dist[h_roi - edge :, :].ravel(),
                dist[:, :edge].ravel(),
                dist[:, w_roi - edge :].ravel(),
            ]
        )
        edge_threshold = float(np.percentile(dist_edges, 95)) if dist_edges.size else 0.0
        threshold = max(8.0, edge_threshold * 1.2)

        mask = (dist > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    color_mask = _background_aware_mask(roi)

    if cv2.countNonZero(color_mask) < 10:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, color_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return x, y, w, h

    all_points = np.concatenate(contours)
    rx, ry, rw, rh = cv2.boundingRect(all_points)

    margin = 2
    rx = max(0, rx - margin)
    ry = max(0, ry - margin)
    rw = min(w - rx, rw + 2 * margin)
    rh = min(h - ry, rh + 2 * margin)

    # Дополнительное обрезание однотонных светлых/тёмных полос по краям,
    # чтобы убирать белые letterbox'ы.
    def _trim_uniform_edges(gray_roi: np.ndarray, max_ratio: float = 0.25) -> Tuple[int, int, int, int]:
        r_h, r_w = gray_roi.shape
        row_mean = gray_roi.mean(axis=1)
        row_std = gray_roi.std(axis=1)
        col_mean = gray_roi.mean(axis=0)
        col_std = gray_roi.std(axis=0)

        std_thresh_row = max(3.0, float(np.percentile(row_std, 15)))
        std_thresh_col = max(3.0, float(np.percentile(col_std, 15)))
        bright_thresh = 242.0
        dark_thresh = 18.0

        def _find_start(arr_mean, arr_std, std_thresh, limit):
            start = 0
            while start < limit and arr_std[start] < std_thresh and (arr_mean[start] > bright_thresh or arr_mean[start] < dark_thresh):
                start += 1
            return start

        def _find_end(arr_mean, arr_std, std_thresh, limit):
            end = len(arr_mean)
            while end > 0 and (len(arr_mean) - end) < limit and arr_std[end - 1] < std_thresh and (arr_mean[end - 1] > bright_thresh or arr_mean[end - 1] < dark_thresh):
                end -= 1
            return end

        max_trim_rows = int(r_h * max_ratio)
        max_trim_cols = int(r_w * max_ratio)

        top = _find_start(row_mean, row_std, std_thresh_row, max_trim_rows)
        bottom = _find_end(row_mean, row_std, std_thresh_row, max_trim_rows)
        left = _find_start(col_mean, col_std, std_thresh_col, max_trim_cols)
        right = _find_end(col_mean, col_std, std_thresh_col, max_trim_cols)

        new_w = max(1, right - left)
        new_h = max(1, bottom - top)
        return left, top, new_w, new_h

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    trim_l, trim_t, trim_w, trim_h = _trim_uniform_edges(gray_roi)

    rx = rx + trim_l
    ry = ry + trim_t
    rw = min(rw - trim_l, trim_w)
    rh = min(rh - trim_t, trim_h)

    # Лёгкий запас по краям, чтобы не съедать полезный контент.
    pad_x = int(0.02 * w)
    pad_y = int(0.02 * h)

    rx = max(0, rx - pad_x)
    ry = max(0, ry - pad_y)
    rw = min(w - rx, rw + 2 * pad_x)
    rh = min(h - ry, rh + 2 * pad_y)

    final_x = x + rx
    final_y = y + ry
    final_w = rw
    final_h = rh

    print(f"[AUTOCROP] Original: {w}x{h}, Refined: {final_w}x{final_h} (Removed top={ry}, bottom={h - (ry + rh)}, left={rx}, right={w - (rx + rw)})")

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
