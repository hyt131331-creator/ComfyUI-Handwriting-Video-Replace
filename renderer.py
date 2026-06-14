from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FIXED_OUTPUT_CUTS = ((630 / 60.0, (645 + 1) / 60.0),)


@dataclass
class RenderOptions:
    line_one: str
    line_two: str
    text_position: str = "bottom-center"
    font_size: int = 36
    font_color: str = "#000000"
    tracking_mode: str = "camera_shake"
    max_seconds: float = 23.0
    text_seconds: float = 10.62
    fade_seconds: float = 0.22
    output_fps: float = 60.0


def render_video(input_path: Path, output_dir: Path, options: RenderOptions) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="handwriting_render_"))
    silent_video = work_dir / "render_no_audio.avi"
    output_path = output_dir / f"result_{uuid.uuid4().hex[:12]}.mp4"

    try:
        _render_frames_to_video(input_path, silent_video, options)
        _mux_audio_if_possible(input_path, silent_video, output_path)
        return output_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _render_frames_to_video(input_path: Path, output_path: Path, options: RenderOptions) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    output_fps = float(options.output_fps or min(source_fps, 24))
    output_fps = max(8.0, min(output_fps, 60.0))
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_duration = source_total_frames / source_fps if source_total_frames else float(options.max_seconds or 0)
    render_seconds = float(options.max_seconds or source_duration or 0)
    if source_duration:
        render_seconds = min(render_seconds, source_duration)
    cut_segments = _active_cut_segments(render_seconds)
    output_seconds = max(0.0, render_seconds - _cut_duration_until(render_seconds, cut_segments))
    max_source_frames = int(render_seconds * source_fps) if render_seconds else source_total_frames
    max_output_frames = int(round(output_seconds * output_fps)) if output_seconds else 0
    text_until_seconds = float(options.text_seconds or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        raise ValueError("Cannot open video writer")

    paper_template = _make_paper_template(options)
    written_frames = 0
    tracker = _PaperTracker()
    source_index = -1
    current_frame = None
    current_quad = None
    current_mask = None
    previous_render_quad_for_fade = None
    text_locked_off = False

    try:
        while max_output_frames == 0 or written_frames < max_output_frames:
            output_time = written_frames / output_fps
            source_time = _source_time_after_cuts(output_time, cut_segments)
            target_source_index = int(source_time * source_fps)
            if max_source_frames and target_source_index >= max_source_frames:
                break

            while source_index < target_source_index:
                ok, frame = cap.read()
                if not ok:
                    current_frame = None
                    break
                source_index += 1
                current_frame = frame
                current_quad, current_mask = tracker.update(frame)
            if current_frame is None:
                break

            show_text = (not text_locked_off) and (not text_until_seconds or source_time <= text_until_seconds)
            fade_progress = _fade_progress(source_time, text_until_seconds, options.fade_seconds)
            if show_text:
                if current_quad is not None and current_mask is not None and _paper_mask_has_red_support(current_frame, current_mask):
                    paper_quad = current_quad.astype(np.float32)
                    paper_mask = current_mask
                    previous_render_quad = previous_render_quad_for_fade.copy() if previous_render_quad_for_fade is not None else None
                    previous_render_quad_for_fade = paper_quad.copy()
                elif not _frame_has_red_paper(current_frame):
                    paper_quad, paper_mask = (None, None)
                    text_locked_off = True
                else:
                    paper_quad, paper_mask = (None, None)
            else:
                paper_quad, paper_mask = (None, None)

            if show_text and paper_quad is not None and paper_mask is not None:
                text_layer = _warp_template_to_frame(paper_template, paper_quad, width, height)
                visible_mask = _visible_paper_mask(current_frame, paper_quad, paper_mask)
                surface_mask = _text_surface_mask(current_frame, visible_mask if visible_mask is not None else paper_mask)
                text_layer_clipped = _clip_layer_to_paper(text_layer, surface_mask)
                text_layer_clipped = _apply_paper_fade(text_layer_clipped, paper_quad, previous_render_quad, fade_progress)
                retained_ratio = _retained_alpha_ratio(text_layer, text_layer_clipped)
                if _layer_visible_enough(text_layer_clipped) and retained_ratio >= 0.12:
                    composed = _ink_blend(current_frame, text_layer_clipped)
                else:
                    composed = current_frame
            else:
                composed = current_frame
            writer.write(composed)
            written_frames += 1
    finally:
        writer.release()
        cap.release()

    if written_frames == 0:
        raise ValueError("No frames were read from video")


def _active_cut_segments(render_seconds: float) -> tuple[tuple[float, float], ...]:
    segments: list[tuple[float, float]] = []
    for start, end in _FIXED_OUTPUT_CUTS:
        if render_seconds <= start:
            continue
        clipped_end = min(end, render_seconds)
        if clipped_end > start:
            segments.append((start, clipped_end))
    return tuple(segments)


def _cut_duration_until(seconds: float, segments: tuple[tuple[float, float], ...]) -> float:
    total = 0.0
    for start, end in segments:
        if seconds <= start:
            break
        total += max(0.0, min(seconds, end) - start)
    return total


def _source_time_after_cuts(output_time: float, segments: tuple[tuple[float, float], ...]) -> float:
    source_time = output_time
    for start, end in segments:
        if source_time < start:
            break
        source_time += end - start
    return source_time


def _visible_paper_mask(frame_bgr: np.ndarray, paper_quad: np.ndarray, paper_mask: np.ndarray) -> np.ndarray | None:
    frame_h, frame_w = frame_bgr.shape[:2]
    paper_area = cv2.countNonZero(paper_mask)
    if paper_area < frame_h * frame_w * 0.025:
        return None

    detected_quad, _ratio, detected_mask = _detect_red_paper(frame_bgr)
    if detected_quad is None or detected_mask is None:
        return None

    detected_area = cv2.countNonZero(detected_mask)
    if detected_area < frame_h * frame_w * 0.025:
        return None

    overlap = cv2.countNonZero(cv2.bitwise_and(detected_mask, paper_mask))
    overlap_ratio = overlap / max(min(detected_area, paper_area), 1)
    if overlap_ratio < 0.04:
        return None

    center_delta = np.linalg.norm(paper_quad.mean(axis=0) - detected_quad.mean(axis=0))
    if center_delta >= max(frame_w, frame_h) * 0.45:
        return None
    return detected_mask


def _detect_render_paper(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    quad, _ratio, mask = _detect_red_paper(frame_bgr)
    if quad is not None and mask is not None:
        return quad, mask
    return _detect_red_paper_loose(frame_bgr)


def _detect_red_paper_loose(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_area = frame_h * frame_w
    mask = _make_red_mask(frame_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < frame_area * 0.025:
        return None, None

    x, y, w, h = cv2.boundingRect(contour)
    if w < frame_w * 0.16 or h < frame_h * 0.16:
        return None, None
    if (w * h) < frame_area * 0.035:
        return None, None

    paper_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(paper_mask, [contour], -1, 255, thickness=cv2.FILLED)
    paper_mask = cv2.erode(paper_mask, np.ones((7, 7), np.uint8), iterations=1)
    quad = _contour_to_quad(contour)
    if quad is None:
        quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    return quad, paper_mask


def _paper_still_visible(frame_bgr: np.ndarray, paper_quad: np.ndarray, paper_mask: np.ndarray) -> bool:
    return _visible_paper_mask(frame_bgr, paper_quad, paper_mask) is not None


def _paper_mask_has_red_support(frame_bgr: np.ndarray, paper_mask: np.ndarray) -> bool:
    paper_area = cv2.countNonZero(paper_mask)
    if paper_area <= 0:
        return False
    red_mask = _make_red_mask(frame_bgr)
    red_on_paper = cv2.countNonZero(cv2.bitwise_and(red_mask, paper_mask))
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    return red_on_paper / paper_area >= 0.035 or red_on_paper >= frame_area * 0.003


def _frame_has_red_paper(frame_bgr: np.ndarray) -> bool:
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    red_mask = _make_red_mask(frame_bgr)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest_area = max(cv2.contourArea(contour) for contour in contours)
    return largest_area >= frame_area * 0.025


def _fallback_visible_paper_mask(frame_bgr: np.ndarray, paper_mask: np.ndarray) -> np.ndarray | None:
    paper_area = cv2.countNonZero(paper_mask)
    if paper_area <= 0:
        return None
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    red_mask = _make_red_mask(frame_bgr)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    red_mask = cv2.dilate(red_mask, np.ones((23, 23), np.uint8), iterations=1)
    overlap = cv2.bitwise_and(red_mask, paper_mask)
    overlap_count = cv2.countNonZero(overlap)
    if overlap_count < max(60, min(paper_area * 0.020, frame_area * 0.003)):
        return None
    return paper_mask


class _PaperTracker:
    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.quad: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.lost_frames = 0
        self.frame_count = 0
        self.ever_initialized = False

    def update(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape[:2]

        if self.prev_gray is None or self.quad is None or self.points is None or len(self.points) < 10:
            if self.ever_initialized:
                self.prev_gray = gray
                self._reset(keep_gray=True)
                return None, None
            quad, _ratio, mask = _detect_red_paper(frame_bgr)
            if quad is None or mask is None:
                self.prev_gray = gray
                self._reset()
                return None, None
            self.quad = quad.astype(np.float32)
            self.points = _find_tracking_points(gray, _quad_to_mask(self.quad, frame_w, frame_h))
            self.prev_gray = gray
            self.lost_frames = 0
            self.frame_count += 1
            self.ever_initialized = True
            return self.quad, _quad_to_mask(self.quad, frame_w, frame_h, erode=13)

        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.points.reshape(-1, 1, 2).astype(np.float32),
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            return self._recover_or_fail(frame_bgr, gray)

        status = status.reshape(-1).astype(bool)
        old_good = self.points.reshape(-1, 2)[status]
        new_good = next_points.reshape(-1, 2)[status]
        if len(old_good) < 8:
            return self._recover_or_fail(frame_bgr, gray)

        matrix, inliers = cv2.findHomography(old_good, new_good, cv2.RANSAC, 3.0)
        if matrix is None or inliers is None or int(inliers.sum()) < 6:
            matrix, inliers = cv2.estimateAffinePartial2D(old_good, new_good, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if matrix is None or inliers is None or int(inliers.sum()) < 6:
                return self._recover_or_fail(frame_bgr, gray)
            affine = np.eye(3, dtype=np.float32)
            affine[:2, :] = matrix.astype(np.float32)
            matrix = affine

        new_quad = cv2.perspectiveTransform(self.quad.reshape(1, 4, 2), matrix.astype(np.float32)).reshape(4, 2)
        if not _valid_quad(new_quad, frame_w, frame_h):
            return self._recover_or_fail(frame_bgr, gray)

        # Use tracked points only. Redetection is reserved for recovery, otherwise the text jumps.
        self.quad = new_quad.astype(np.float32)
        inlier_mask = inliers.reshape(-1).astype(bool)
        self.points = new_good[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
        paper_mask = _quad_to_mask(self.quad, frame_w, frame_h, erode=13)
        if len(self.points) < 35 or self.frame_count % 12 == 0:
            fresh = _find_tracking_points(gray, paper_mask)
            if fresh is not None and len(fresh) >= 10:
                self.points = fresh
        self.prev_gray = gray
        self.lost_frames = 0
        self.frame_count += 1
        return self.quad, paper_mask

    def _recover_or_fail(self, frame_bgr: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        self.lost_frames += 1
        quad, _ratio, _mask = _detect_red_paper(frame_bgr)
        if quad is None:
            self.prev_gray = gray
            self._reset(keep_gray=True)
            return None, None
        frame_h, frame_w = gray.shape[:2]
        if self.quad is not None and not _plausible_recovery_quad(quad, self.quad, frame_w, frame_h):
            self.prev_gray = gray
            self._reset(keep_gray=True)
            return None, None
        self.quad = quad.astype(np.float32)
        paper_mask = _quad_to_mask(self.quad, frame_w, frame_h, erode=13)
        self.points = _find_tracking_points(gray, paper_mask)
        self.prev_gray = gray
        self.lost_frames = 0
        self.frame_count += 1
        return self.quad, paper_mask

    def _reset(self, keep_gray: bool = False) -> None:
        if not keep_gray:
            self.prev_gray = None
        self.quad = None
        self.points = None
        self.lost_frames = 0


def _make_paper_template(options: RenderOptions) -> np.ndarray:
    paper_w, paper_h = 720, 1120
    layer = Image.new("RGBA", (paper_w, paper_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    template_font_size = max(30, min(int(options.font_size * 1.7), 96))
    font = _load_font(template_font_size)
    lines = [options.line_one.strip(), options.line_two.strip()]
    lines = [line for line in lines if line]
    if not lines:
        lines = [" "]

    line_gap = max(8, template_font_size // 5)
    bboxes = [draw.textbbox((0, 0), line, font=font, stroke_width=0) for line in lines]
    text_w = max(b[2] - b[0] for b in bboxes)
    text_h = sum(b[3] - b[1] for b in bboxes) + line_gap * (len(lines) - 1)
    x, y = _paper_position_to_xy(options.text_position, paper_w, paper_h, text_w, text_h, template_font_size)
    fill = _hex_to_rgba(options.font_color)

    cursor_y = y
    for line, bbox in zip(lines, bboxes):
        line_w = bbox[2] - bbox[0]
        draw.text(
            (x + (text_w - line_w) / 2, cursor_y),
            line,
            font=font,
            fill=fill,
            stroke_width=0,
            stroke_fill=(0, 0, 0, 0),
        )
        cursor_y += (bbox[3] - bbox[1]) + line_gap

    bgra = cv2.cvtColor(np.array(layer), cv2.COLOR_RGBA2BGRA)
    alpha = bgra[:, :, 3]
    if alpha.max() > 0:
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.45)
        texture = np.random.default_rng(7).normal(1.0, 0.075, size=alpha.shape)
        alpha = np.clip(alpha.astype(np.float32) * texture * 0.80, 0, 188).astype(np.uint8)
        bgra[:, :, 3] = alpha
    return bgra


def _detect_red_paper(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    mask = _make_red_mask(frame_bgr)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, None

    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    ratio = float(area / frame_area)
    if ratio < 0.07:
        return None, ratio, None

    x, y, w, h = cv2.boundingRect(contour)
    frame_h, frame_w = frame_bgr.shape[:2]
    box_ratio = (w * h) / float(frame_w * frame_h)
    aspect = h / max(w, 1)
    if box_ratio < 0.09:
        return None, ratio, None
    if w < frame_w * 0.22 or h < frame_h * 0.24:
        return None, ratio, None
    if not (0.55 <= aspect <= 3.20):
        return None, ratio, None

    paper_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(paper_mask, [contour], -1, 255, thickness=cv2.FILLED)
    paper_mask = cv2.erode(paper_mask, np.ones((15, 15), np.uint8), iterations=1)
    quad = _contour_to_quad(contour)
    if quad is None:
        quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    return quad, ratio, paper_mask


def _make_red_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 40, 55], dtype=np.uint8)
    upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([165, 35, 55], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)
    lower_pink = np.array([145, 30, 60], dtype=np.uint8)
    upper_pink = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask |= cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask |= cv2.inRange(hsv, lower_pink, upper_pink)
    return mask


def _find_tracking_points(gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=180,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
        mask=mask,
    )
    if points is None or len(points) < 8:
        return None
    return points.astype(np.float32)


def _quad_to_mask(quad: np.ndarray, width: int, height: int, erode: int = 0) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    if erode > 0:
        mask = cv2.erode(mask, np.ones((erode, erode), np.uint8), iterations=1)
    return mask


def _valid_quad(quad: np.ndarray, width: int, height: int) -> bool:
    if not np.all(np.isfinite(quad)):
        return False
    margin = max(width, height) * 0.35
    if (quad[:, 0] < -margin).any() or (quad[:, 0] > width + margin).any():
        return False
    if (quad[:, 1] < -margin).any() or (quad[:, 1] > height + margin).any():
        return False
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    frame_area = float(width * height)
    if area < frame_area * 0.04 or area > frame_area * 1.35:
        return False
    edge_top = np.linalg.norm(quad[1] - quad[0])
    edge_bottom = np.linalg.norm(quad[2] - quad[3])
    edge_left = np.linalg.norm(quad[3] - quad[0])
    edge_right = np.linalg.norm(quad[2] - quad[1])
    if min(edge_top, edge_bottom, edge_left, edge_right) < 20:
        return False
    aspect = (edge_left + edge_right) / max(edge_top + edge_bottom, 1.0)
    return 0.9 <= aspect <= 2.8


def _plausible_recovery_quad(new_quad: np.ndarray, previous_quad: np.ndarray, width: int, height: int) -> bool:
    frame_area = float(width * height)
    new_area = abs(cv2.contourArea(new_quad.astype(np.float32)))
    prev_area = abs(cv2.contourArea(previous_quad.astype(np.float32)))
    if new_area > frame_area * 0.92:
        return False
    if prev_area > 1 and new_area / prev_area > 1.75:
        return False
    center_delta = np.linalg.norm(new_quad.mean(axis=0) - previous_quad.mean(axis=0))
    if center_delta > max(width, height) * 0.22:
        return False
    return True


def _render_quad_change_ok(new_quad: np.ndarray, previous_quad: np.ndarray, width: int, height: int) -> bool:
    prev_area = abs(cv2.contourArea(previous_quad.astype(np.float32)))
    new_area = abs(cv2.contourArea(new_quad.astype(np.float32)))
    if prev_area <= 1 or new_area <= 1:
        return False
    area_ratio = new_area / prev_area
    if area_ratio < 0.72 or area_ratio > 1.28:
        return False
    center_delta = np.linalg.norm(new_quad.mean(axis=0) - previous_quad.mean(axis=0))
    if center_delta > max(width, height) * 0.055:
        return False
    corner_delta = np.mean(np.linalg.norm(new_quad - previous_quad, axis=1))
    if corner_delta > max(width, height) * 0.075:
        return False
    return True


def _smooth_render_quad(
    new_quad: np.ndarray,
    previous_quad: np.ndarray | None,
    width: int,
    height: int,
    frame_bgr: np.ndarray,
    current_mask: np.ndarray,
) -> np.ndarray | None:
    if previous_quad is None:
        return new_quad.astype(np.float32)

    if _render_quad_change_ok(new_quad, previous_quad, width, height):
        alpha = 1.0
    elif _broad_render_quad_change_ok(new_quad, previous_quad, width, height) and _paper_mask_has_red_support(frame_bgr, current_mask):
        alpha = 0.82
    elif _paper_mask_has_red_support(frame_bgr, current_mask) and _valid_quad(new_quad, width, height):
        return new_quad.astype(np.float32)
    else:
        return None

    smoothed = previous_quad.astype(np.float32) * (1.0 - alpha) + new_quad.astype(np.float32) * alpha
    if not _valid_quad(smoothed, width, height):
        return None
    return smoothed


def _broad_render_quad_change_ok(new_quad: np.ndarray, previous_quad: np.ndarray, width: int, height: int) -> bool:
    prev_area = abs(cv2.contourArea(previous_quad.astype(np.float32)))
    new_area = abs(cv2.contourArea(new_quad.astype(np.float32)))
    if prev_area <= 1 or new_area <= 1:
        return False
    area_ratio = new_area / prev_area
    if area_ratio < 0.58 or area_ratio > 1.58:
        return False
    center_delta = np.linalg.norm(new_quad.mean(axis=0) - previous_quad.mean(axis=0))
    if center_delta > max(width, height) * 0.16:
        return False
    corner_delta = np.mean(np.linalg.norm(new_quad - previous_quad, axis=1))
    if corner_delta > max(width, height) * 0.20:
        return False
    return True


def _contour_to_quad(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    for factor in (0.018, 0.025, 0.035, 0.05):
        approx = cv2.approxPolyDP(contour, factor * perimeter, True)
        if len(approx) == 4:
            return _order_quad(approx.reshape(4, 2).astype(np.float32))

    hull = cv2.convexHull(contour)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    if cv2.contourArea(box) <= 1:
        return None
    return _order_quad(box)


def _order_quad(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _warp_template_to_frame(template_bgra: np.ndarray, paper_quad: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = template_bgra.shape[:2]
    src_quad = np.array([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_quad, paper_quad.astype(np.float32))
    warped = cv2.warpPerspective(
        template_bgra,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    if warped[:, :, 3].max() > 0:
        warped[:, :, 3] = cv2.GaussianBlur(warped[:, :, 3], (3, 3), 0.35)
    return warped


def _clip_layer_to_paper(layer_bgra: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    clipped = layer_bgra.copy()
    clipped[:, :, 3] = cv2.bitwise_and(clipped[:, :, 3], paper_mask)
    return clipped


def _text_surface_mask(frame_bgr: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    strict = cv2.inRange(hsv, np.array([0, 105, 120], dtype=np.uint8), np.array([14, 255, 255], dtype=np.uint8))
    strict |= cv2.inRange(hsv, np.array([160, 95, 120], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    strict |= cv2.inRange(hsv, np.array([145, 95, 125], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    strict = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    strict = cv2.dilate(strict, np.ones((17, 17), np.uint8), iterations=1)
    surface = cv2.bitwise_and(strict, paper_mask)
    if cv2.countNonZero(surface) < cv2.countNonZero(paper_mask) * 0.35:
        return paper_mask
    return surface


def _layer_visible_enough(layer_bgra: np.ndarray) -> bool:
    alpha = layer_bgra[:, :, 3]
    return cv2.countNonZero(alpha) >= 60 and int(alpha.max()) >= 18


def _fade_progress(seconds: float, text_until_seconds: float, fade_seconds: float) -> float:
    fade_seconds = max(0.0, float(fade_seconds or 0.0))
    if not text_until_seconds or fade_seconds <= 0:
        return 0.0
    start = text_until_seconds - fade_seconds
    if seconds <= start:
        return 0.0
    if seconds >= text_until_seconds:
        return 1.0
    return max(0.0, min(1.0, (seconds - start) / fade_seconds))


def _apply_paper_fade(
    layer_bgra: np.ndarray,
    paper_quad: np.ndarray | None,
    previous_quad: np.ndarray | None,
    progress: float,
) -> np.ndarray:
    if progress <= 0:
        return layer_bgra
    adjusted = layer_bgra.copy()
    direction = np.array([1.0, 0.25], dtype=np.float32)
    if paper_quad is not None and previous_quad is not None:
        delta = paper_quad.mean(axis=0) - previous_quad.mean(axis=0)
        norm = float(np.linalg.norm(delta))
        if norm > 0.1:
            direction = delta.astype(np.float32) / norm

    length = 3 + int(round(progress * 16))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    for i in range(length):
        offset = i - center
        x = int(round(center + direction[0] * offset))
        y = int(round(center + direction[1] * offset))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    if kernel.sum() <= 0:
        kernel[center, center] = 1.0
    kernel /= kernel.sum()
    adjusted[:, :, 3] = cv2.filter2D(adjusted[:, :, 3], -1, kernel)
    opacity = (1.0 - progress) ** 1.6
    adjusted[:, :, 3] = np.clip(adjusted[:, :, 3].astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    return adjusted


def _frame_blur_score(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _retained_alpha_ratio(original_bgra: np.ndarray, clipped_bgra: np.ndarray) -> float:
    original = cv2.countNonZero(original_bgra[:, :, 3])
    if original <= 0:
        return 0.0
    clipped = cv2.countNonZero(clipped_bgra[:, :, 3])
    return clipped / original


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    project_font = Path(__file__).resolve().parent / "assets" / "aaa.TTF"
    if not project_font.exists():
        raise FileNotFoundError(f"Font file not found: {project_font}")
    return ImageFont.truetype(str(project_font), size=size)


def _paper_position_to_xy(position: str, width: int, height: int, text_w: int, text_h: int, margin: int) -> tuple[int, int]:
    pos = (position or "bottom-center").lower().replace("_", "-")
    x = int((width - text_w) * 0.50)

    if "top" in pos:
        y = int(height * 0.28)
    elif "center" in pos and "bottom" not in pos:
        y = int(height * 0.43)
    else:
        y = int(height * 0.53)
    return max(0, int(x)), max(0, int(y))


def _hex_to_rgba(value: str) -> tuple[int, int, int, int]:
    text = (value or "#FFFFFF").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        text = "FFFFFF"
    r = int(text[0:2], 16)
    g = int(text[2:4], 16)
    b = int(text[4:6], 16)
    return r, g, b, 235


def _alpha_blend(frame_bgr: np.ndarray, layer_bgra: np.ndarray) -> np.ndarray:
    alpha = layer_bgra[:, :, 3:4].astype(np.float32) / 255.0
    text_bgr = layer_bgra[:, :, :3].astype(np.float32)
    base = frame_bgr.astype(np.float32)
    blended = base * (1.0 - alpha) + text_bgr * alpha
    return blended.astype(np.uint8)


def _ink_blend(frame_bgr: np.ndarray, layer_bgra: np.ndarray) -> np.ndarray:
    alpha_u8 = layer_bgra[:, :, 3]
    if int(alpha_u8.max()) <= 0:
        return frame_bgr

    ys, xs = np.nonzero(alpha_u8)
    if len(xs) == 0:
        return frame_bgr
    pad = 4
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(frame_bgr.shape[0], int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(frame_bgr.shape[1], int(xs.max()) + pad + 1)

    result = frame_bgr.copy()
    frame_roi = frame_bgr[y1:y2, x1:x2]
    layer_roi = layer_bgra[y1:y2, x1:x2]
    alpha = layer_roi[:, :, 3].astype(np.float32) / 255.0
    if float(alpha.max()) <= 0:
        return frame_bgr

    paper_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    paper_gray = cv2.GaussianBlur(paper_gray, (0, 0), 1.1)
    local_texture = np.clip(0.82 + paper_gray * 0.34, 0.72, 1.08)
    ink_alpha = np.clip(alpha * local_texture, 0.0, 0.76)[:, :, None]

    base = frame_roi.astype(np.float32)
    ink = layer_roi[:, :, :3].astype(np.float32)
    darkened_paper = base * (1.0 - ink_alpha * 0.72)
    tinted_ink = ink * 0.22 + base * 0.10
    blended = darkened_paper * (1.0 - ink_alpha * 0.18) + tinted_ink * (ink_alpha * 0.18)
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _mux_audio_if_possible(input_path: Path, silent_video: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create browser-compatible MP4 output")

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
