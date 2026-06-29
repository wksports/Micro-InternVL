"""Evaluation script for Micro-InternVL (official InternVL3.5-4B based)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_internvl.dataset import EMDS7COCODataset, collate_fn, load_base_novel_split, load_category_names
from micro_internvl.model_wrapper import MicroInternVL
from micro_internvl.queries import HierarchicalQuerySet
from micro_internvl.utils import apply_nms, box_cxcywh_to_xyxy, setup_logging

logger = logging.getLogger(__name__)


def load_coco(coco_path: Path) -> COCO:
    return COCO(str(coco_path))


def scale_coco_areas_for_input_resolution(coco_gt: COCO, imgsz: int) -> COCO:
    """Scale annotation areas to input resolution for COCOeval small/medium/large thresholds."""
    dataset = {
        "images": list(coco_gt.dataset["images"]),
        "categories": list(coco_gt.dataset["categories"]),
        "annotations": [],
    }
    img_id_to_size = {im["id"]: (im["width"], im["height"]) for im in coco_gt.dataset["images"]}

    for ann in coco_gt.dataset["annotations"]:
        x, y, w, h = ann["bbox"]
        img_w, img_h = img_id_to_size[ann["image_id"]]
        scale_x = imgsz / img_w
        scale_y = imgsz / img_h
        scaled_area = (w * scale_x) * (h * scale_y)
        new_ann = dict(ann)
        new_ann["area"] = float(scaled_area)
        dataset["annotations"].append(new_ann)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dataset, f)
        tmp_path = f.name

    return COCO(tmp_path)


def micro_internvl_predictions_to_coco(
    model: MicroInternVL,
    dataloader: DataLoader,
    idx_to_coco_id: Dict[int, int],
    device: torch.device,
    confidence_threshold: float = 0.001,
    nms_threshold: float = 0.5,
    top_k: int = 100,
) -> List[Dict[str, Any]]:
    """Run inference and return COCO-format predictions."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(pixel_values=images)

            pred_boxes = outputs["pred_boxes"]
            pred_logits = outputs["pred_logits"]
            pred_objectness = outputs["pred_objectness"]

            B = pred_boxes.size(0)
            for i in range(B):
                image_id = int(targets[i]["image_id"].item())
                orig_h, orig_w = targets[i]["orig_size"].tolist()

                boxes = pred_boxes[i]
                logits = pred_logits[i]
                obj_scores = pred_objectness[i, :, 0].sigmoid()

                cls_scores, labels = logits.sigmoid().max(dim=-1)
                scores = obj_scores * cls_scores

                keep_mask = scores > confidence_threshold
                boxes = boxes[keep_mask]
                scores = scores[keep_mask]
                labels = labels[keep_mask]

                if len(boxes) == 0:
                    continue

                boxes_xyxy = box_cxcywh_to_xyxy(boxes)
                keep = apply_nms(boxes_xyxy, scores, iou_threshold=nms_threshold, top_k=top_k)

                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                boxes_abs = boxes.cpu().numpy()
                boxes_abs[:, [0, 2]] *= orig_w
                boxes_abs[:, [1, 3]] *= orig_h

                for j in range(len(boxes)):
                    coco_cat_id = idx_to_coco_id[int(labels[j].item())]
                    x, y, w, h = boxes_abs[j]
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(coco_cat_id),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(scores[j].item()),
                    })

    return predictions


def evaluate_coco(coco_gt: COCO, predictions: List[Dict[str, Any]], cat_ids: List[int] | None = None) -> Dict[str, float]:
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if cat_ids is not None:
        coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Micro-InternVL")
    parser.add_argument("--config", type=str, default="micro_internvl/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_dir = Path(args.config).parent
    log_dir = (base_dir / config["logging"].get("log_dir", "../outputs/logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / f"eval_{args.split}.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    category_map_path = (base_dir / config["data"]["category_map"]).resolve()
    category_names, coco_to_idx, idx_to_name = load_category_names(str(category_map_path))
    num_classes = len(idx_to_name)
    idx_to_coco_id = {v: k for k, v in coco_to_idx.items()}

    model = MicroInternVL.from_pretrained(
        args.checkpoint,
        micro_internvl_config=config["micro_internvl"],
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
        device_map=config["model"].get("device_map", None),
    )
    model.to(device)
    model.eval()

    query_file = config["data"].get("query_file")
    query_path = (base_dir / query_file).resolve() if query_file else None
    if query_path and query_path.exists():
        query_set = HierarchicalQuerySet.from_file(str(query_path))
        category_name_list = [idx_to_name[i] for i in range(num_classes)]
        queries = query_set.get_queries(category_name_list, level=None)
        logger.info("Using hierarchical queries from query file")
    else:
        queries = [idx_to_name[i] for i in range(num_classes)]
        logger.info("Using category name queries")
    model.set_text_embeddings(model.encode_text_queries(queries))

    split_json = (base_dir / config["data"][f"{args.split}_json"]).resolve()
    image_dir = (base_dir / config["data"]["image_dir"]).resolve()
    dataset = EMDS7COCODataset(
        annotation_file=str(split_json),
        image_dir=str(image_dir),
        category_map=str(category_map_path),
        resolution=config["data"]["resolution"],
        split=args.split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        collate_fn=collate_fn,
    )

    coco_gt_raw = load_coco(split_json)
    coco_gt = scale_coco_areas_for_input_resolution(coco_gt_raw, config["data"]["resolution"])

    predictions = micro_internvl_predictions_to_coco(
        model=model,
        dataloader=dataloader,
        idx_to_coco_id=idx_to_coco_id,
        device=device,
        confidence_threshold=config["inference"].get("confidence_threshold", 0.001),
        nms_threshold=config["inference"].get("nms_threshold", 0.5),
        top_k=config["inference"].get("top_k", 100),
    )
    logger.info(f"Generated {len(predictions)} predictions")

    all_metrics = evaluate_coco(coco_gt, predictions)

    base_novel_split_path = (base_dir / config["data"]["base_novel_split"]).resolve()
    _, novel_ids = load_base_novel_split(str(base_novel_split_path))
    novel_metrics = evaluate_coco(coco_gt, predictions, cat_ids=novel_ids)

    result = {
        "model": "Micro-InternVL",
        "split": args.split,
        "checkpoint": args.checkpoint,
        "num_predictions": len(predictions),
        "AP": all_metrics["AP"],
        "AP50": all_metrics["AP50"],
        "AP75": all_metrics["AP75"],
        "AP_small": all_metrics["AP_small"],
        "AP_medium": all_metrics["AP_medium"],
        "AP_large": all_metrics["AP_large"],
        "AP_novel": novel_metrics["AP"],
    }

    print(json.dumps(result, indent=2))

    if args.out is None:
        out_dir = base_dir.parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = out_dir / f"micro_internvl_{args.split}_metrics.json"

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
