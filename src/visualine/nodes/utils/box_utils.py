from typing import Any, Iterable, List, Tuple

import numpy as np
import torch


def box_to_numpy(box: Any) -> np.ndarray:
    if isinstance(box, torch.Tensor):
        arr = box.detach().cpu().numpy()
    else:
        arr = np.asarray(box)

    return arr.reshape(-1).astype(np.float32)


def convert_box_to_xyxy(
    box: Any,
    width: int,
    height: int,
    box_format: str = "xyxy_abs",
) -> Tuple[int, int, int, int]:
    """
    Convert a box to absolute pixel xyxy format.

    Supported formats:
    - xyxy_abs:   [x1, y1, x2, y2] in pixels
    - xyxy_norm:  [x1, y1, x2, y2] in 0-1 normalized coords
    - cxcywh_abs: [cx, cy, w, h] in pixels
    - cxcywh_norm:[cx, cy, w, h] in 0-1 normalized coords
    - auto:       best-effort detection
    """
    arr = box_to_numpy(box)

    if arr.size < 4:
        return 0, 0, 0, 0

    a, b, c, d = arr[:4]
    fmt = box_format.lower().strip()

    if fmt == "auto":
        max_v = float(np.max(np.abs(arr[:4])))

        # If all values are <= 1.5, assume normalized GroundingDINO cxcywh.
        # Raw GroundingDINO boxes are normally cxcywh_norm.
        if max_v <= 1.5:
            fmt = "cxcywh_norm"
        else:
            fmt = "xyxy_abs"

    if fmt == "xyxy_abs":
        x1, y1, x2, y2 = a, b, c, d

    elif fmt == "xyxy_norm":
        x1, y1, x2, y2 = a * width, b * height, c * width, d * height

    elif fmt == "cxcywh_abs":
        cx, cy, bw, bh = a, b, c, d
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

    elif fmt == "cxcywh_norm":
        cx, cy, bw, bh = a * width, b * height, c * width, d * height
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

    else:
        raise ValueError(f"Unsupported box format: {box_format}")

    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))

    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0

    return x1, y1, x2, y2


def expand_box_xyxy(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    x1 = int(max(0, min(width - 1, round(x1 - bw * expand_ratio))))
    y1 = int(max(0, min(height - 1, round(y1 - bh * expand_ratio))))
    x2 = int(max(0, min(width, round(x2 + bw * expand_ratio))))
    y2 = int(max(0, min(height, round(y2 + bh * expand_ratio))))

    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0

    return x1, y1, x2, y2


def filter_boxes_xyxy(
    boxes: Iterable[Tuple[int, int, int, int]],
    width: int,
    height: int,
    min_area_ratio: float = 0.0001,
    max_area_ratio: float = 0.65,
    max_boxes: int = 50,
) -> List[Tuple[int, int, int, int]]:
    frame_area = float(width * height)
    valid = []

    for box in boxes:
        x1, y1, x2, y2 = box

        if x2 <= x1 or y2 <= y1:
            continue

        area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area

        if area_ratio < min_area_ratio:
            continue

        # Critical: prevents one huge detection from blurring the whole frame.
        if area_ratio > max_area_ratio:
            continue

        valid.append((x1, y1, x2, y2))

    return valid[:max_boxes]


def get_boxes_from_data(data: dict) -> list:
    """
    Preferred key is boxes_xyxy.
    Fallback to boxes for older nodes.
    """
    if not isinstance(data, dict):
        return []

    return data.get("boxes_xyxy", data.get("boxes", []))