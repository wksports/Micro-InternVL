"""Training script for Micro-InternVL (official InternVL3.5-4B based)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_internvl.dataset import EMDS7COCODataset, collate_fn, load_base_novel_split, load_category_names
from micro_internvl.evaluate import (
    evaluate_coco,
    load_coco,
    micro_internvl_predictions_to_coco,
    scale_coco_areas_for_input_resolution,
)
from micro_internvl.losses import BoxTextAlignmentLoss, DetectionLoss, PatchTextAlignmentLoss
from micro_internvl.model_wrapper import MicroInternVL
from micro_internvl.queries import HierarchicalQuerySet
from micro_internvl.utils import setup_logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(config_path: str, path: str) -> str:
    """Resolve a path relative to the config file's directory."""
    if os.path.isabs(path):
        return path
    base_dir = Path(config_path).parent
    return str((base_dir / path).resolve())


def load_query_set(
    config: Dict[str, Any],
    config_path: str,
    idx_to_name: Dict[int, str],
) -> Tuple[List[str], Optional[HierarchicalQuerySet]]:
    """Load text queries and hard negatives from query file if available."""
    query_file = config["data"].get("query_file")
    if query_file:
        query_file = resolve_path(config_path, query_file)
    if query_file and Path(query_file).exists():
        query_set = HierarchicalQuerySet.from_file(query_file)
        category_names = [idx_to_name[i] for i in range(len(idx_to_name))]
        default_queries = query_set.get_queries(category_names, level=None)
        return default_queries, query_set

    default_queries = [name for _, name in sorted(idx_to_name.items())]
    return default_queries, None


def build_optimizer(model: MicroInternVL, config: Dict[str, Any]) -> AdamW:
    """Build AdamW with separate LR for detection head and LoRA parameters."""
    head_params = []
    lora_params = []
    other_params = []

    for name, param in model.base_model.named_parameters():
        if not param.requires_grad:
            continue
        if "detection_head" in name:
            head_params.append(param)
        elif "lora_" in name.lower():
            lora_params.append(param)
        else:
            other_params.append(param)

    logger.info(
        f"Optimizer groups: head={len(head_params)}, lora={len(lora_params)}, other={len(other_params)}"
    )

    param_groups = [
        {"params": head_params, "lr": config["training"]["head_lr"]},
        {"params": lora_params, "lr": config["training"]["lora_lr"]},
    ]
    if len(other_params) > 0:
        param_groups.append({"params": other_params, "lr": config["training"]["lora_lr"]})

    optimizer = AdamW(
        param_groups,
        weight_decay=config["training"].get("weight_decay", 0.05),
    )
    return optimizer


def build_scheduler(optimizer: AdamW, config: Dict[str, Any], num_training_steps: int, num_warmup_steps: int):
    """Build cosine scheduler with linear warmup."""
    if num_warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_training_steps - num_warmup_steps),
            eta_min=0.0,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[num_warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=0.0)
    return scheduler


def get_amp_settings(mixed_precision: str, device: torch.device) -> Tuple[bool, torch.dtype, bool]:
    """Return (use_amp, amp_dtype, use_grad_scaler)."""
    if device.type != "cuda":
        return False, torch.float32, False

    precision = str(mixed_precision).lower()
    if precision in {"no", "none", "false", "fp32", "float32"}:
        return False, torch.float32, False
    if precision in {"bf16", "bfloat16"}:
        return True, torch.bfloat16, False
    if precision in {"fp16", "float16"}:
        return True, torch.float16, True
    raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")


class EarlyStopping:
    """Early stopping based on a validation metric."""

    def __init__(
        self,
        patience: int,
        metric: str,
        mode: str = "max",
        min_delta: float = 0.0,
        save_best: bool = True,
    ) -> None:
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.save_best = save_best
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.best_ckpt_path: Optional[str] = None

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def step(self, value: float, epoch: int) -> Tuple[bool, bool]:
        """Return (improved, should_stop)."""
        if self._is_better(value):
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return True, False
        self.counter += 1
        return False, self.counter >= self.patience


def run_validation(
    model: MicroInternVL,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    idx_to_coco_id: Dict[int, int],
    val_json: str,
    base_novel_split_path: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    """Run validation and return COCO metrics."""
    predictions = micro_internvl_predictions_to_coco(
        model=model,
        dataloader=val_loader,
        idx_to_coco_id=idx_to_coco_id,
        device=device,
        confidence_threshold=config["inference"].get("confidence_threshold", 0.001),
        nms_threshold=config["inference"].get("nms_threshold", 0.5),
        top_k=config["inference"].get("top_k", 100),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    coco_gt_raw = load_coco(Path(val_json))
    coco_gt = scale_coco_areas_for_input_resolution(coco_gt_raw, config["data"]["resolution"])
    all_metrics = evaluate_coco(coco_gt, predictions)

    _, novel_ids = load_base_novel_split(str(base_novel_split_path))
    novel_metrics = evaluate_coco(coco_gt, predictions, cat_ids=novel_ids)

    return {
        **all_metrics,
        "AP_novel": novel_metrics["AP"],
        "num_predictions": len(predictions),
    }


def train_one_epoch(
    model: MicroInternVL,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    detection_loss_fn: DetectionLoss,
    patch_text_loss_fn: Optional[PatchTextAlignmentLoss],
    box_text_loss_fn: Optional[BoxTextAlignmentLoss],
    text_embeddings: torch.Tensor,
    hard_negative_embeddings: Optional[torch.Tensor],
    config: Dict[str, Any],
    epoch: int,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_grad_scaler: bool,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_grad_scaler else None

    total_loss = 0.0
    total_loss_det = 0.0
    total_loss_objectness = 0.0
    total_loss_patch = 0.0
    total_loss_box = 0.0
    num_batches = 0

    lambda_patch = config["loss"].get("lambda_patch", 0.0)
    lambda_box = config["loss"].get("lambda_box", 0.0)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(pbar):
        images = images.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            outputs = model(
                pixel_values=images,
                text_queries=None,
                return_patch_features=(lambda_patch > 0 or lambda_box > 0),
            )

            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]
            pred_objectness = outputs["pred_objectness"]

            target_labels = [t["labels"] for t in targets]
            target_boxes = [t["boxes"] for t in targets]
            ignore_boxes = [t.get("ignore_boxes", torch.zeros(0, 4)) for t in targets]

            loss_dict = detection_loss_fn(
                pred_logits,
                pred_boxes,
                pred_objectness,
                target_labels,
                target_boxes,
                ignore_boxes,
            )
            loss = loss_dict["loss_det"]

            if lambda_patch > 0 and patch_text_loss_fn is not None and "patch_features" in outputs:
                loss_patch = patch_text_loss_fn(
                    outputs["patch_features"],
                    text_embeddings,
                    target_boxes,
                    target_labels,
                    hard_negative_embeddings,
                )
                loss = loss + lambda_patch * loss_patch
                total_loss_patch += loss_patch.item()

            if lambda_box > 0 and box_text_loss_fn is not None and "patch_features" in outputs:
                loss_box = box_text_loss_fn(
                    outputs["patch_features"],
                    pred_boxes,
                    text_embeddings,
                    target_boxes,
                    target_labels,
                    hard_negative_embeddings,
                )
                loss = loss + lambda_box * loss_box
                total_loss_box += loss_box.item()

        loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        total_loss_det += loss_dict["loss_det"].item()
        total_loss_objectness += loss_dict["loss_objectness"].item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum_steps:.4f}",
            "det": f"{loss_dict['loss_det'].item():.4f}",
            "obj": f"{loss_dict['loss_objectness'].item():.4f}",
        })

    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_det": total_loss_det / max(num_batches, 1),
        "loss_objectness": total_loss_objectness / max(num_batches, 1),
        "loss_patch": total_loss_patch / max(num_batches, 1),
        "loss_box": total_loss_box / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Micro-InternVL")
    parser.add_argument("--config", type=str, default="micro_internvl/config.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    output_dir = Path(resolve_path(args.config, config["training"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(resolve_path(args.config, config["logging"].get("log_dir", "outputs/logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / "train.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config}")

    category_map_path = resolve_path(args.config, config["data"]["category_map"])
    category_names, coco_to_idx, idx_to_name = load_category_names(category_map_path)
    num_classes = len(idx_to_name)
    idx_to_coco_id = {v: k for k, v in coco_to_idx.items()}
    logger.info(f"Number of classes: {num_classes}")

    queries, query_set = load_query_set(config, args.config, idx_to_name)
    logger.info(f"Loaded {len(queries)} text queries")

    train_json = resolve_path(args.config, config["data"]["train_json"])
    val_json = resolve_path(args.config, config["data"]["val_json"])
    image_dir = resolve_path(args.config, config["data"]["image_dir"])
    base_novel_split = resolve_path(args.config, config["data"]["base_novel_split"])

    train_dataset = EMDS7COCODataset(
        annotation_file=train_json,
        image_dir=image_dir,
        category_map=category_map_path,
        resolution=config["data"]["resolution"],
        split="train",
        base_novel_split=base_novel_split,
        use_base_only=True,
    )
    val_dataset = EMDS7COCODataset(
        annotation_file=val_json,
        image_dir=image_dir,
        category_map=category_map_path,
        resolution=config["data"]["resolution"],
        split="val",
        base_novel_split=base_novel_split,
        use_base_only=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        collate_fn=collate_fn,
    )

    # Build model. use_backbone_lora is injected via the official InternVLChatConfig.
    lora_cfg = config["lora"]
    model = MicroInternVL(
        model_path=config["model"]["base_model"],
        micro_internvl_config=config["micro_internvl"],
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
        device_map=config["model"].get("device_map", None),
    )

    # Apply official backbone LoRA
    model.base_model.wrap_backbone_lora(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
    )

    # Freeze language model
    for param in model.base_model.language_model.parameters():
        param.requires_grad = False

    model.to(device)

    # Pre-compute text embeddings
    model.set_text_embeddings(model.encode_text_queries(queries))
    text_embeddings = model.text_embeddings

    hard_negative_embeddings = None
    if query_set is not None and len(query_set.get_hard_negatives()) > 0:
        with torch.no_grad():
            hard_negative_embeddings = model.encode_text_queries(query_set.get_hard_negatives())
        logger.info(f"Loaded {len(query_set.get_hard_negatives())} hard negatives")

    # Enable gradient checkpointing on vision model
    if config["training"].get("gradient_checkpointing", False):
        if hasattr(model.base_model.vision_model, "gradient_checkpointing_enable"):
            model.base_model.vision_model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on vision model")
        else:
            logger.warning("Vision model does not support gradient_checkpointing_enable")

    detection_loss_fn = DetectionLoss(
        num_classes=num_classes,
        weight_class=config["loss"].get("class_weight", 1.0),
        weight_bbox=config["loss"].get("bbox_weight", 5.0),
        weight_giou=config["loss"].get("giou_weight", 2.0),
        weight_objectness=config["loss"].get("objectness_weight", 1.0),
        focal_alpha=config["loss"].get("focal_alpha", 0.25),
        focal_gamma=config["loss"].get("focal_gamma", 2.0),
        ignore_iou_threshold=config["loss"].get("ignore_iou_threshold", 0.3),
    )

    patch_text_loss_fn = None
    box_text_loss_fn = None
    if config["loss"].get("lambda_patch", 0.0) > 0:
        patch_text_loss_fn = PatchTextAlignmentLoss(
            temperature=config["micro_internvl"].get("temperature", 0.07)
        )
    if config["loss"].get("lambda_box", 0.0) > 0:
        box_text_loss_fn = BoxTextAlignmentLoss(
            temperature=config["micro_internvl"].get("temperature", 0.07),
            grid_size=32,
        )

    optimizer = build_optimizer(model, config)

    num_epochs = config["training"]["num_epochs"]
    steps_per_epoch = len(train_loader)
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    optimizer_steps_per_epoch = max(1, math.ceil(steps_per_epoch / grad_accum_steps))
    num_training_steps = num_epochs * optimizer_steps_per_epoch
    warmup_epochs = config["training"].get("warmup_epochs", 2)
    num_warmup_steps = warmup_epochs * optimizer_steps_per_epoch

    scheduler = build_scheduler(optimizer, config, num_training_steps, num_warmup_steps)

    use_amp, amp_dtype, use_grad_scaler = get_amp_settings(
        config["training"].get("mixed_precision", "no"),
        device,
    )

    es_cfg = config["training"].get("early_stopping")
    early_stopper: Optional[EarlyStopping] = None
    if es_cfg and es_cfg.get("patience", 0) > 0:
        early_stopper = EarlyStopping(
            patience=es_cfg["patience"],
            metric=es_cfg.get("metric", "AP"),
            mode=es_cfg.get("mode", "max"),
            min_delta=es_cfg.get("min_delta", 0.0),
            save_best=es_cfg.get("save_best", True),
        )
        logger.info(
            f"Early stopping enabled: metric={early_stopper.metric}, "
            f"patience={early_stopper.patience}, mode={early_stopper.mode}"
        )

    start_epoch = 0
    if args.resume:
        logger.warning("Resume not yet implemented")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            detection_loss_fn=detection_loss_fn,
            patch_text_loss_fn=patch_text_loss_fn,
            box_text_loss_fn=box_text_loss_fn,
            text_embeddings=text_embeddings,
            hard_negative_embeddings=hard_negative_embeddings,
            config=config,
            epoch=epoch + 1,
            device=device,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=config["training"].get("max_grad_norm", 0.1),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_grad_scaler=use_grad_scaler,
        )

        logger.info(f"Train metrics: {train_metrics}")

        eval_every = config["training"].get("eval_every_epochs", 1)
        should_stop = False
        if (epoch + 1) % eval_every == 0:
            val_metrics = run_validation(
                model=model,
                val_loader=val_loader,
                config=config,
                device=device,
                idx_to_coco_id=idx_to_coco_id,
                val_json=val_json,
                base_novel_split_path=base_novel_split,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            logger.info(f"Validation metrics: {val_metrics}")

            if early_stopper is not None:
                metric_value = val_metrics[early_stopper.metric]
                improved, should_stop = early_stopper.step(metric_value, epoch + 1)
                if improved and early_stopper.save_best:
                    best_ckpt_dir = output_dir / "checkpoint-best"
                    if best_ckpt_dir.exists():
                        shutil.rmtree(best_ckpt_dir)
                    model.save_pretrained(str(best_ckpt_dir))
                    early_stopper.best_ckpt_path = str(best_ckpt_dir)
                    logger.info(
                        f"New best {early_stopper.metric}={metric_value:.4f} at epoch {epoch + 1}, "
                        f"saved to {best_ckpt_dir}"
                    )
                elif should_stop:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best {early_stopper.metric}={early_stopper.best_value:.4f} "
                        f"at epoch {early_stopper.best_epoch}."
                    )

        if (epoch + 1) % config["training"].get("save_every_epochs", 5) == 0:
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
            model.save_pretrained(str(ckpt_dir))
            logger.info(f"Saved checkpoint to {ckpt_dir}")

        if should_stop:
            break

    if early_stopper is not None and early_stopper.best_ckpt_path and early_stopper.save_best:
        final_dir = output_dir / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(early_stopper.best_ckpt_path, final_dir)
        logger.info(f"Copied best checkpoint to {final_dir}")
    else:
        final_dir = output_dir / "final"
        model.save_pretrained(str(final_dir))
        logger.info(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
