"""Losses for Micro-InternVL.

Includes DETR-style Hungarian matcher, focal + L1 + GIoU detection loss,
and patch-text / box-text InfoNCE contrastive alignment losses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    linear_sum_assignment = None


def greedy_bipartite_matching(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy bipartite matching as a fallback when scipy is unavailable.

    Iteratively picks the lowest-cost (pred, tgt) pair until all targets are matched.
    Not optimal but sufficient for smoke tests and light usage.
    """
    cost = cost.clone()
    num_preds, num_tgts = cost.shape
    pred_indices = []
    tgt_indices = []

    for _ in range(min(num_preds, num_tgts)):
        min_val = cost.min()
        if min_val.item() == float("inf"):
            break
        flat_idx = cost.view(-1).argmin()
        pred_idx = int(flat_idx // num_tgts)
        tgt_idx = int(flat_idx % num_tgts)
        pred_indices.append(pred_idx)
        tgt_indices.append(tgt_idx)
        cost[pred_idx, :] = float("inf")
        cost[:, tgt_idx] = float("inf")

    return torch.tensor(pred_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute generalized IoU between two sets of boxes in xyxy format.

    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]

    Returns:
        giou: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    # Smallest enclosing box
    lt_min = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_max = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_max = (rb_max - lt_min).clamp(min=0)
    area_c = wh_max[:, :, 0] * wh_max[:, :, 1]

    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes in xyxy format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    """DETR-style bipartite matching between predictions and ground-truth boxes."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Match predictions to targets.

        Args:
            pred_logits: [B, 1024, num_classes]
            pred_boxes: [B, 1024, 4]
            target_labels: List of [num_targets_i] label tensors
            target_boxes: List of [num_targets_i, 4] box tensors

        Returns:
            List of (pred_indices, target_indices) tuples, one per image.
        """
        pred_logits = torch.nan_to_num(pred_logits.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        pred_boxes = torch.nan_to_num(pred_boxes.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        target_boxes = [
            torch.nan_to_num(boxes.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            for boxes in target_boxes
        ]
        bs, num_queries, num_classes = pred_logits.shape
        pred_probs = pred_logits.sigmoid()

        indices = []
        for i in range(bs):
            tgt_labels = target_labels[i].to(pred_logits.device)
            tgt_boxes = target_boxes[i]
            num_targets = len(tgt_labels)
            if num_targets == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue

            # Classification cost: negative probability of correct class
            cost_class = -pred_probs[i][:, tgt_labels]  # [1024, num_targets]

            # L1 box cost
            pred_box_i = pred_boxes[i]  # [1024, 4]
            tgt_box_i = tgt_boxes.to(pred_box_i.device)
            cost_bbox = torch.cdist(pred_box_i, tgt_box_i, p=1)  # [1024, num_targets]

            # GIoU cost
            pred_xyxy = box_cxcywh_to_xyxy(pred_box_i)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_box_i)
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)  # [1024, num_targets]

            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
            cost = cost.cpu()
            if HAS_SCIPY:
                pred_idx, tgt_idx = linear_sum_assignment(cost)
                indices.append((torch.from_numpy(pred_idx).long(), torch.from_numpy(tgt_idx).long()))
            else:
                pred_idx, tgt_idx = greedy_bipartite_matching(cost)
                indices.append((pred_idx, tgt_idx))

        return indices


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for classification."""
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


class DetectionLoss(nn.Module):
    """Detection loss: focal classification + objectness + L1 + GIoU."""

    def __init__(
        self,
        num_classes: int,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_objectness: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ignore_iou_threshold: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_objectness = weight_objectness
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ignore_iou_threshold = ignore_iou_threshold
        self.matcher = HungarianMatcher(
            cost_class=weight_class,
            cost_bbox=weight_bbox,
            cost_giou=weight_giou,
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_objectness: Optional[torch.Tensor],
        target_labels: List[torch.Tensor],
        target_boxes: List[torch.Tensor],
        ignore_boxes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute detection loss.

        Args:
            pred_logits: [B, 1024, num_classes]
            pred_boxes: [B, 1024, 4]
            pred_objectness: Optional [B, 1024, 1] objectness logits
            target_labels: List of [num_targets_i]
            target_boxes: List of [num_targets_i, 4]
            ignore_boxes: Optional list of novel boxes to mask from background loss

        Returns:
            Dict with loss_class, loss_objectness, loss_bbox, loss_giou, loss_det.
        """
        bs, num_queries, _ = pred_logits.shape
        device = pred_logits.device
        pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=50.0, neginf=-50.0)
        pred_boxes = torch.nan_to_num(pred_boxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if pred_objectness is not None:
            pred_objectness = torch.nan_to_num(pred_objectness, nan=0.0, posinf=50.0, neginf=-50.0)

        # Match predictions to targets
        indices = self.matcher(pred_logits, pred_boxes, target_labels, target_boxes)

        # Build target classification one-hot
        target_classes = torch.full(
            (bs, num_queries), self.num_classes, dtype=torch.long, device=device
        )  # background class index = num_classes
        src_logits = pred_logits
        target_objectness = torch.zeros((bs, num_queries, 1), dtype=pred_logits.dtype, device=device)
        positive_mask = torch.zeros((bs, num_queries), dtype=torch.bool, device=device)

        src_boxes_list = []
        target_boxes_list = []
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            pred_idx_device = pred_idx.to(device)
            target_classes[i, pred_idx_device] = target_labels[i][tgt_idx].to(device)
            target_objectness[i, pred_idx_device, 0] = 1.0
            positive_mask[i, pred_idx_device] = True
            src_boxes_list.append(pred_boxes[i, pred_idx_device])
            target_boxes_list.append(target_boxes[i][tgt_idx].to(device))

        # Classification loss (focal)
        target_classes_onehot = torch.zeros(
            bs, num_queries, self.num_classes + 1, device=device
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]  # remove background class

        loss_class_raw = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        ).sum(dim=-1)

        valid_query_mask = torch.ones((bs, num_queries), dtype=pred_logits.dtype, device=device)
        if ignore_boxes is not None:
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            for i, img_ignore_boxes in enumerate(ignore_boxes):
                if img_ignore_boxes.numel() == 0:
                    continue
                ignore_xyxy = box_cxcywh_to_xyxy(img_ignore_boxes.to(device))
                max_iou = box_iou(pred_xyxy[i], ignore_xyxy).max(dim=1).values
                ignored = (max_iou > self.ignore_iou_threshold) & (~positive_mask[i])
                valid_query_mask[i, ignored] = 0.0

        valid_denominator = valid_query_mask.sum().clamp(min=1.0)
        loss_class = (loss_class_raw * valid_query_mask).sum() / valid_denominator

        if pred_objectness is not None:
            if pred_objectness.dim() == 2:
                pred_objectness = pred_objectness.unsqueeze(-1)
            loss_objectness_raw = F.binary_cross_entropy_with_logits(
                pred_objectness,
                target_objectness,
                reduction="none",
            ).squeeze(-1)
            loss_objectness = (loss_objectness_raw * valid_query_mask).sum() / valid_denominator
        else:
            loss_objectness = torch.tensor(0.0, device=device)

        # Box regression losses
        if len(src_boxes_list) > 0:
            src_boxes_cat = torch.cat(src_boxes_list, dim=0)
            target_boxes_cat = torch.cat(target_boxes_list, dim=0)

            loss_bbox = F.l1_loss(src_boxes_cat, target_boxes_cat, reduction="sum")
            loss_giou = (1 - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes_cat),
                    box_cxcywh_to_xyxy(target_boxes_cat),
                )
            )).sum()

            num_total_targets = sum(len(t) for t in target_labels)
            loss_bbox = loss_bbox / num_total_targets
            loss_giou = loss_giou / num_total_targets
        else:
            loss_bbox = torch.tensor(0.0, device=device)
            loss_giou = torch.tensor(0.0, device=device)

        loss_det = (
            self.weight_class * loss_class
            + self.weight_objectness * loss_objectness
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou
        )

        return {
            "loss_class": loss_class,
            "loss_objectness": loss_objectness,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_det": loss_det,
        }


class PatchTextAlignmentLoss(nn.Module):
    """InfoNCE loss aligning GT-box interior patches with text embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        patch_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
        hard_negative_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute patch-text alignment loss.

        Args:
            patch_features: [B, 1024, D]
            text_embeddings: [K, D] category text embeddings
            target_boxes: List of [num_targets_i, 4] boxes in normalized cxcywh
            target_labels: List of [num_targets_i] label indices
            hard_negative_embeddings: [N_neg, D] optional hard negative embeddings

        Returns:
            Scalar loss.
        """
        device = patch_features.device
        B, num_patches, D = patch_features.shape
        grid_size = int(num_patches ** 0.5)
        assert grid_size * grid_size == num_patches, "patch_features must be a square grid"

        # Normalize features
        patch_features = F.normalize(patch_features, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Build patch coordinate grid [grid_size, grid_size, 2] in [0, 1]
        y_coords = torch.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size, device=device)
        x_coords = torch.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size, device=device)
        gy, gx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        patch_centers = torch.stack([gx, gy], dim=-1)  # [H, W, 2]
        patch_centers = patch_centers.reshape(num_patches, 2)  # [1024, 2]

        losses = []
        for i in range(B):
            boxes = target_boxes[i].to(device)  # [T, 4]
            labels = target_labels[i].to(device)  # [T]
            if len(boxes) == 0:
                continue

            # Convert boxes to xyxy
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)  # [T, 4]

            # Determine which patches fall inside each GT box
            centers = patch_centers[:, None, :]  # [1024, 1, 2]
            boxes_expanded = boxes_xyxy[None, :, :]  # [1, T, 4]
            inside = (
                (centers[..., 0] >= boxes_expanded[..., 0])
                & (centers[..., 0] <= boxes_expanded[..., 2])
                & (centers[..., 1] >= boxes_expanded[..., 1])
                & (centers[..., 1] <= boxes_expanded[..., 3])
            )  # [1024, T]

            # For each GT box, collect positive patch features
            for t in range(len(boxes)):
                pos_mask = inside[:, t]
                if pos_mask.sum() == 0:
                    continue
                pos_patches = patch_features[i, pos_mask]  # [N_pos, D]
                pos_label = labels[t].item()

                # Positive text embedding
                pos_text = text_embeddings[pos_label: pos_label + 1]  # [1, D]

                # Negative text embeddings: all categories except pos_label + hard negatives
                neg_mask = torch.ones(text_embeddings.size(0), dtype=torch.bool, device=device)
                neg_mask[pos_label] = False
                neg_text = text_embeddings[neg_mask]  # [K-1, D]

                if hard_negative_embeddings is not None:
                    hard_neg = F.normalize(hard_negative_embeddings, dim=-1).to(device)
                    neg_text = torch.cat([neg_text, hard_neg], dim=0)

                # Compute similarity [N_pos, 1 + num_neg]
                all_text = torch.cat([pos_text, neg_text], dim=0)  # [1 + num_neg, D]
                logits = torch.matmul(pos_patches, all_text.T) / self.temperature  # [N_pos, 1 + num_neg]
                labels_pos = torch.zeros(pos_patches.size(0), dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, labels_pos)
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()


class BoxTextAlignmentLoss(nn.Module):
    """InfoNCE loss aligning RoI-pooled box features with matched text embeddings."""

    def __init__(self, temperature: float = 0.07, grid_size: int = 32):
        super().__init__()
        self.temperature = temperature
        self.grid_size = grid_size

    def _roi_pool_patch_features(
        self,
        patch_features: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """RoI-pool patch features for each box using average pooling over covered patches.

        Args:
            patch_features: [B, 1024, D]
            boxes: [B, T, 4] normalized cxcywh boxes

        Returns:
            box_features: [N, D] where N is total number of boxes across batch.
        """
        B, num_patches, D = patch_features.shape
        H = W = self.grid_size
        assert num_patches == H * W

        device = patch_features.device
        patch_features_grid = patch_features.reshape(B, H, W, D)

        box_features_list = []
        for b in range(B):
            img_boxes = boxes[b]
            if len(img_boxes) == 0:
                continue
            img_boxes = img_boxes.to(device)

            # Convert cxcywh to grid indices
            cx, cy, w, h = img_boxes.unbind(-1)
            x1 = (cx - 0.5 * w).clamp(0, 1)
            y1 = (cy - 0.5 * h).clamp(0, 1)
            x2 = (cx + 0.5 * w).clamp(0, 1)
            y2 = (cy + 0.5 * h).clamp(0, 1)

            x1_idx = (x1 * W).long().clamp(0, W - 1)
            y1_idx = (y1 * H).long().clamp(0, H - 1)
            x2_idx = (x2 * W).long().clamp(0, W - 1)
            y2_idx = (y2 * H).long().clamp(0, H - 1)

            for t in range(len(img_boxes)):
                x1_t, y1_t, x2_t, y2_t = x1_idx[t].item(), y1_idx[t].item(), x2_idx[t].item(), y2_idx[t].item()
                if x2_t < x1_t or y2_t < y1_t:
                    box_feat = patch_features_grid[b, y1_t, x1_t]
                else:
                    region = patch_features_grid[b, y1_t:y2_t+1, x1_t:x2_t+1]
                    box_feat = region.mean(dim=[0, 1])
                box_features_list.append(box_feat)

        if len(box_features_list) == 0:
            return torch.zeros(0, D, device=device)
        return torch.stack(box_features_list, dim=0)

    def forward(
        self,
        patch_features: torch.Tensor,
        pred_boxes: torch.Tensor,
        text_embeddings: torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
        hard_negative_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute box-text alignment loss.

        Args:
            patch_features: [B, 1024, D]
            pred_boxes: [B, 1024, 4]
            text_embeddings: [K, D]
            target_boxes: List of [num_targets_i, 4]
            target_labels: List of [num_targets_i]
            hard_negative_embeddings: [N_neg, D] optional

        Returns:
            Scalar loss.
        """
        device = patch_features.device
        B = patch_features.size(0)

        # For box-text alignment, we use matched predicted boxes.
        # In the simplest form, match each GT box to the best predicted box
        # (via center distance) and pool its patch region.
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)

        # We need classification logits to call matcher; build dummy logits
        # using target label similarity.
        dummy_logits = torch.zeros(
            B, pred_boxes.size(1), text_embeddings.size(0), device=device
        )
        for i in range(B):
            for label in target_labels[i]:
                dummy_logits[i, :, label] = 1.0

        indices = matcher(dummy_logits, pred_boxes, target_labels, target_boxes)

        box_features_list = []
        labels_list = []
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_boxes = pred_boxes[i, pred_idx]  # [T, 4]
            # Pool per-box features (treat all as one image with B=1)
            box_feat = self._roi_pool_patch_features(
                patch_features[i:i+1], matched_boxes.unsqueeze(0)
            )
            if box_feat.size(0) > 0:
                box_features_list.append(box_feat)
                labels_list.append(target_labels[i][tgt_idx].to(device))

        if len(box_features_list) == 0:
            return torch.tensor(0.0, device=device)

        box_features = torch.cat(box_features_list, dim=0)  # [N, D]
        labels = torch.cat(labels_list, dim=0)  # [N]

        # Normalize
        box_features = F.normalize(box_features, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Similarities [N, K]
        logits = torch.matmul(box_features, text_embeddings.T) / self.temperature

        # Cross-entropy with GT labels
        loss = F.cross_entropy(logits, labels)
        return loss
