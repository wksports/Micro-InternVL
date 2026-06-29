"""EMDS-7 COCO-format dataset loader for Micro-InternVL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Default InternVL normalization (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def box_xywh_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert (x, y, w, h) to (cx, cy, w, h) and normalize by image size."""
    cx = boxes[:, 0] + boxes[:, 2] / 2.0
    cy = boxes[:, 1] + boxes[:, 3] / 2.0
    w = boxes[:, 2]
    h = boxes[:, 3]
    return np.stack([cx, cy, w, h], axis=1)


def load_category_names(category_map_path: str) -> Tuple[List[str], Dict[int, int], Dict[int, str]]:
    """Load category mapping from EMDS-7 category_map.json.

    Returns:
        category_names: List of category names indexed by COCO category id (1-based).
        coco_to_idx: Mapping from COCO category id to 0-based model class index.
        idx_to_name: Mapping from 0-based model class index to category name.
    """
    with open(category_map_path, "r") as f:
        category_map = json.load(f)

    # category_map is {group_id: {id: coco_id, name: category_name}}
    items = []
    for group_id, info in category_map.items():
        items.append((int(info["id"]), info["name"]))

    # Sort by COCO id
    items.sort(key=lambda x: x[0])
    max_id = max(x[0] for x in items)

    category_names = [""] * (max_id + 1)
    for cid, name in items:
        category_names[cid] = name

    coco_to_idx = {cid: i for i, (cid, _) in enumerate(items)}
    idx_to_name = {i: name for i, (_, name) in enumerate(items)}

    return category_names, coco_to_idx, idx_to_name


def load_base_novel_split(split_path: str) -> Tuple[List[int], List[int]]:
    """Load base/novel category IDs (COCO 1-based ids)."""
    with open(split_path, "r") as f:
        split = json.load(f)
    return split["base"], split["novel"]


class EMDS7COCODataset(Dataset):
    """PyTorch Dataset for EMDS-7 COCO-format annotations."""

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        category_map: str,
        resolution: int = 448,
        split: str = "train",
        base_novel_split: Optional[str] = None,
        use_base_only: bool = False,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
    ):
        super().__init__()
        self.annotation_file = annotation_file
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.split = split
        self.image_mean = np.array(image_mean or IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.image_std = np.array(image_std or IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

        # Load annotations
        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = self.coco_data["annotations"]

        # Build image_id -> annotations mapping
        self.image_to_annotations: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.image_to_annotations.setdefault(img_id, []).append(ann)
        self.image_to_ignore_annotations: Dict[int, List[Dict[str, Any]]] = {}

        self.image_ids = sorted(self.images.keys())

        # Category mapping
        self.category_names, self.coco_to_idx, self.idx_to_name = load_category_names(category_map)
        self.num_classes = len(self.coco_to_idx)

        # Base/novel filtering
        self.base_ids = set()
        self.novel_ids = set()
        if base_novel_split:
            base_coco_ids, novel_coco_ids = load_base_novel_split(base_novel_split)
            self.base_ids = set(base_coco_ids)
            self.novel_ids = set(novel_coco_ids)

            if use_base_only and split == "train":
                # Filter annotations and images to only include base categories
                self._filter_to_base()

        logger.info(
            f"EMDS7 {split} dataset: {len(self.image_ids)} images, "
            f"{sum(len(v) for v in self.image_to_annotations.values())} annotations, "
            f"{self.num_classes} classes"
        )

    def _filter_to_base(self) -> None:
        """Remove novel categories from training data."""
        new_image_to_annotations = {}
        new_image_ids = []

        for img_id in self.image_ids:
            anns = self.image_to_annotations.get(img_id, [])
            base_anns = [ann for ann in anns if ann["category_id"] in self.base_ids]
            novel_anns = [ann for ann in anns if ann["category_id"] in self.novel_ids]
            if len(base_anns) > 0:
                new_image_to_annotations[img_id] = base_anns
                self.image_to_ignore_annotations[img_id] = novel_anns
                new_image_ids.append(img_id)

        self.image_to_annotations = new_image_to_annotations
        self.image_ids = sorted(new_image_ids)
        logger.info(f"Filtered to base categories: {len(self.image_ids)} images remain.")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, img_info: Dict[str, Any]) -> np.ndarray:
        """Load and resize image to target resolution."""
        img_path = self.image_dir / img_info["file_name"]
        if not img_path.exists():
            # Try basename only
            img_path = self.image_dir / Path(img_info["file_name"]).name

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        return np.array(image, dtype=np.float32) / 255.0

    def _normalize(self, image: np.ndarray) -> torch.Tensor:
        """Normalize image and convert to CHW tensor."""
        image = (image - self.image_mean) / self.image_std
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(image).float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        anns = self.image_to_annotations.get(img_id, [])
        ignore_anns = self.image_to_ignore_annotations.get(img_id, [])

        # Load image
        image = self._load_image(img_info)
        orig_h, orig_w = img_info.get("height", image.shape[0]), img_info.get("width", image.shape[1])

        # Build boxes and labels
        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: top-left x, y, width, height (absolute)

            # Normalize by image size
            x1 = x / orig_w
            y1 = y / orig_h
            w_norm = w / orig_w
            h_norm = h / orig_h

            # Convert to cxcywh
            cx = x1 + w_norm / 2.0
            cy = y1 + h_norm / 2.0

            boxes.append([cx, cy, w_norm, h_norm])
            labels.append(self.coco_to_idx[ann["category_id"]])
            areas.append(ann.get("area", w * h))

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            areas = np.zeros((0,), dtype=np.float32)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            areas = np.array(areas, dtype=np.float32)

        ignore_boxes = []
        for ann in ignore_anns:
            x, y, w, h = ann["bbox"]
            x1 = x / orig_w
            y1 = y / orig_h
            w_norm = w / orig_w
            h_norm = h / orig_h
            ignore_boxes.append([x1 + w_norm / 2.0, y1 + h_norm / 2.0, w_norm, h_norm])
        if len(ignore_boxes) == 0:
            ignore_boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            ignore_boxes = np.array(ignore_boxes, dtype=np.float32)

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "areas": torch.from_numpy(areas),
            "ignore_boxes": torch.from_numpy(ignore_boxes),
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

        image_tensor = self._normalize(image)
        return image_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Collate function for variable-length targets."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
