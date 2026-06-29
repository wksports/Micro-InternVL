"""Training script for Micro-InternVL (official InternVL3.5-4B based)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
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

from micro_internvl.dataset import EMDS7COCODataset, collate_fn, load_category_names
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


def load_query_set(config: Dict[str, Any], idx_to_name: Dict[int, str]) -> Tuple[List[str], Optional[HierarchicalQuerySet]]:
    """Load text queries and hard negatives from query file if available."""
    query_file = config["data"].get("query_file")
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
            T_max=num_training_steps - num_warmup_steps,
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


def train_one_epoch(
    model: MicroInternVL,
    dataloader: DataLoader,
    optimizer: AdamW,
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
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_loss = 0.0
    total_loss_det = 0.0
    total_loss_patch = 0.0
    total_loss_box = 0.0
    num_batches = 0

    lambda_patch = config["loss"].get("lambda_patch", 0.0)
    lambda_box = config["loss"].get("lambda_box", 0.0)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(pbar):
        images = images.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(
                pixel_values=images,
                text_queries=None,
                return_patch_features=(lambda_patch > 0 or lambda_box > 0),
            )

            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]

            target_labels = [t["labels"] for t in targets]
            target_boxes = [t["boxes"] for t in targets]

            loss_dict = detection_loss_fn(pred_logits, pred_boxes, target_labels, target_boxes)
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

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        total_loss_det += loss_dict["loss_det"].item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum_steps:.4f}",
            "det": f"{loss_dict['loss_det'].item():.4f}",
        })

    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_det": total_loss_det / max(num_batches, 1),
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
    logger.info(f"Number of classes: {num_classes}")

    queries, query_set = load_query_set(config, idx_to_name)
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
        focal_alpha=config["loss"].get("focal_alpha", 0.25),
        focal_gamma=config["loss"].get("focal_gamma", 2.0),
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
    num_training_steps = num_epochs * steps_per_epoch
    warmup_epochs = config["training"].get("warmup_epochs", 2)
    num_warmup_steps = warmup_epochs * steps_per_epoch

    scheduler = build_scheduler(optimizer, config, num_training_steps, num_warmup_steps)

    use_amp = config["training"].get("mixed_precision", "no") in ["bf16", "bfloat16", "fp16"]

    start_epoch = 0
    if args.resume:
        logger.warning("Resume not yet implemented")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            detection_loss_fn=detection_loss_fn,
            patch_text_loss_fn=patch_text_loss_fn,
            box_text_loss_fn=box_text_loss_fn,
            text_embeddings=text_embeddings,
            hard_negative_embeddings=hard_negative_embeddings,
            config=config,
            epoch=epoch + 1,
            device=device,
            grad_accum_steps=config["training"]["gradient_accumulation_steps"],
            max_grad_norm=config["training"].get("max_grad_norm", 0.1),
            use_amp=use_amp,
        )

        logger.info(f"Train metrics: {train_metrics}")
        scheduler.step()

        if (epoch + 1) % config["training"].get("save_every_epochs", 5) == 0:
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
            model.save_pretrained(str(ckpt_dir))
            logger.info(f"Saved checkpoint to {ckpt_dir}")

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    logger.info(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
