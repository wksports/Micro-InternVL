"""Utility functions for Micro-InternVL."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Union

import torch
import torchvision.ops as ops


def setup_logging(log_file: Union[str, Path], level: int = logging.INFO) -> None:
    """Setup file + console logging."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    top_k: int = 100,
) -> torch.Tensor:
    """Apply non-maximum suppression.

    Args:
        boxes: [N, 4] in xyxy format
        scores: [N]
        iou_threshold: NMS IoU threshold
        top_k: maximum number of detections to keep

    Returns:
        keep_indices: [M] indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    keep = ops.nms(boxes, scores, iou_threshold)
    if top_k > 0:
        keep = keep[:top_k]
    return keep


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)
