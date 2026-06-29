#!/usr/bin/env python3
"""Smoke tests for Micro-InternVL components.

These tests do not require downloading the full InternVL3.5-4B model.
Run with: python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from internvl.model.internvl_chat import MicroInternVLDetectionHead
from micro_internvl.losses import DetectionLoss, PatchTextAlignmentLoss, BoxTextAlignmentLoss


def test_detection_head():
    head = MicroInternVLDetectionHead(input_dim=768, hidden_dim=512, num_layers=2)
    x = torch.randn(2, 1024, 768)
    out = head(x)
    assert out["pred_boxes"].shape == (2, 1024, 4)
    assert out["pred_objectness"].shape == (2, 1024, 1)
    print("PASS: Detection head output shapes correct")


def test_detection_loss():
    loss_fn = DetectionLoss(num_classes=41)
    pred_logits = torch.randn(2, 1024, 41)
    pred_boxes = torch.rand(2, 1024, 4)
    target_labels = [
        torch.tensor([0, 5, 10]),
        torch.tensor([1, 2]),
    ]
    target_boxes = [
        torch.rand(3, 4),
        torch.rand(2, 4),
    ]
    losses = loss_fn(pred_logits, pred_boxes, target_labels, target_boxes)
    assert "loss_det" in losses
    assert losses["loss_det"].item() >= 0
    print("PASS: Detection loss runs")


def test_patch_text_loss():
    loss_fn = PatchTextAlignmentLoss(temperature=0.07)
    patch_features = torch.randn(2, 1024, 768)
    text_embeddings = torch.randn(41, 768)
    target_boxes = [
        torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]),
        torch.tensor([[0.3, 0.3, 0.15, 0.15]]),
    ]
    target_labels = [
        torch.tensor([0, 5]),
        torch.tensor([10]),
    ]
    loss = loss_fn(patch_features, text_embeddings, target_boxes, target_labels)
    assert loss.item() >= 0
    print("PASS: Patch-text alignment loss runs")


def test_box_text_loss():
    loss_fn = BoxTextAlignmentLoss(temperature=0.07)
    patch_features = torch.randn(2, 1024, 768)
    pred_boxes = torch.rand(2, 1024, 4)
    text_embeddings = torch.randn(41, 768)
    target_boxes = [
        torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]),
        torch.tensor([[0.3, 0.3, 0.15, 0.15]]),
    ]
    target_labels = [
        torch.tensor([0, 5]),
        torch.tensor([10]),
    ]
    loss = loss_fn(patch_features, pred_boxes, text_embeddings, target_boxes, target_labels)
    assert loss.item() >= 0
    print("PASS: Box-text alignment loss runs")


if __name__ == "__main__":
    test_detection_head()
    test_detection_loss()
    test_patch_text_loss()
    test_box_text_loss()
    print("\nAll smoke tests passed!")
