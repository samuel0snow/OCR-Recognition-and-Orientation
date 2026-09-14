"""Reusable OCR keyword recognition and annotation functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path: Path) -> np.ndarray:
    """Read an image, including from paths containing Chinese characters."""
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在或不是文件: {path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片（格式损坏或不受支持）: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image safely when its path contains Chinese characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    if suffix not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("输出格式仅支持 .jpg/.jpeg/.png/.bmp/.webp")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法编码输出图片: {path}")
    encoded.tofile(str(path))


def create_ocr() -> PaddleOCR:
    """Create an OCR engine using the inference models bundled with the project."""
    model_dirs = {name: PROJECT_ROOT / "models" / name for name in ("det", "rec", "cls")}
    missing = [
        str(directory)
        for directory in model_dirs.values()
        if not all(
            (directory / filename).is_file()
            for filename in ("inference.pdmodel", "inference.pdiparams")
        )
    ]
    if missing:
        raise FileNotFoundError("本地 OCR 模型不完整，缺少: " + ", ".join(missing))
    return PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        use_gpu=False,
        show_log=False,
        det_model_dir=str(model_dirs["det"]),
        rec_model_dir=str(model_dirs["rec"]),
        cls_model_dir=str(model_dirs["cls"]),
        det_limit_type="max",
        det_limit_side_len=1920,
    )


def normalise(text: str) -> str:
    return "".join(text.casefold().split())


def substring_box(line_box: Any, start: int, end: int, text_length: int) -> list[list[float]]:
    """Interpolate a keyword quadrilateral from an OCR text-line quadrilateral."""
    if text_length <= 0:
        return [[float(x), float(y)] for x, y in line_box]
    points = np.asarray(line_box, dtype=np.float32)
    top_left, top_right, bottom_right, bottom_left = points
    start_ratio, end_ratio = start / text_length, end / text_length

    def point_at(left: np.ndarray, right: np.ndarray, ratio: float) -> np.ndarray:
        return left + (right - left) * ratio

    return [
        point_at(top_left, top_right, start_ratio).tolist(),
        point_at(top_left, top_right, end_ratio).tolist(),
        point_at(bottom_left, bottom_right, end_ratio).tolist(),
        point_at(bottom_left, bottom_right, start_ratio).tolist(),
    ]


def match_text(lines: list[Any], query: str, threshold: int) -> list[dict[str, Any]]:
    """Find every exact occurrence, falling back to line-level fuzzy matches."""
    query = normalise(query)
    exact_matches: list[dict[str, Any]] = []
    fuzzy_candidates: list[dict[str, Any]] = []
    for box, (text, confidence) in lines:
        candidate = normalise(text)
        start = candidate.find(query)
        while start >= 0:
            end = start + len(query)
            exact_matches.append(
                {
                    "text": query,
                    "line_text": text,
                    "confidence": round(float(confidence), 4),
                    "match_score": 100.0,
                    "box": [
                        [round(float(x), 1), round(float(y), 1)]
                        for x, y in substring_box(box, start, end, len(candidate))
                    ],
                }
            )
            start = candidate.find(query, start + 1)

        score = float(fuzz.partial_ratio(query, candidate))
        if score >= threshold:
            fuzzy_candidates.append(
                {
                    "text": text,
                    "line_text": text,
                    "confidence": round(float(confidence), 4),
                    "match_score": round(score, 2),
                    "approximate": True,
                    "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box],
                }
            )
    return exact_matches or fuzzy_candidates


def draw_boxes(image: np.ndarray, matches: list[dict[str, Any]]) -> np.ndarray:
    annotated = image.copy()
    for item in matches:
        points = np.asarray(item["box"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [points], True, (0, 0, 255), 3, cv2.LINE_AA)
    return annotated


def recognize_and_mark(
    ocr: PaddleOCR, image_path: Path, query: str, threshold: int = 85
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Recognize one image and return annotated pixels, matches and line count."""
    if not query.strip():
        raise ValueError("关键词不能为空")
    if not 0 <= threshold <= 100:
        raise ValueError("匹配阈值必须在 0 到 100 之间")
    image = read_image(image_path)
    result = ocr.ocr(image, cls=True)
    lines = result[0] if result and result[0] else []
    matches = match_text(lines, query, threshold)
    return draw_boxes(image, matches), matches, len(lines)
