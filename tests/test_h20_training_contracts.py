#!/usr/bin/env python3
"""Static contracts for H20 training readiness.

These checks intentionally avoid importing torch so they can run in a minimal
environment before the H20 CUDA stack is installed.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class H20TrainingContracts(unittest.TestCase):
    def test_configs_resolve_repo_data_paths(self) -> None:
        for config_name in ["config.yaml", "config_h20.yaml"]:
            text = read_text(f"micro_internvl/{config_name}")
            for value in re.findall(r':\s*"([^"]+)"', text):
                if value.startswith("../data/"):
                    self.assertTrue(
                        (ROOT / "micro_internvl" / value).resolve().exists()
                        or value.endswith("queries.json"),
                        f"{config_name} path does not resolve: {value}",
                    )
            self.assertIn('train_json: "../data/emds7/instances_train.json"', text)
            self.assertIn('val_json: "../data/emds7/instances_val.json"', text)
            self.assertIn('test_json: "../data/emds7/instances_test.json"', text)
            self.assertIn('category_map: "../data/emds7/category_map.json"', text)
            self.assertIn('base_novel_split: "../data/emds7/base_novel_split.json"', text)
            self.assertIn('query_file: "../data/emds7/queries.json"', text)

    def test_internvl35_qwen3_is_supported(self) -> None:
        config_source = read_text("internvl/model/internvl_chat/configuration_internvl_chat.py")
        model_source = read_text("internvl/model/internvl_chat/modeling_internvl_chat.py")

        self.assertIn("Qwen3Config", config_source)
        self.assertIn("Qwen3ForCausalLM", config_source)
        self.assertIn("Qwen3ForCausalLM", model_source)
        self.assertIn("Qwen3DecoderLayer", model_source)

    def test_default_internvl_config_has_supported_llm_architecture(self) -> None:
        config_source = read_text("internvl/model/internvl_chat/configuration_internvl_chat.py")

        self.assertNotIn("llm_config = {'architectures': ['']}", config_source)
        self.assertIn("llm_config = {'architectures': ['LlamaForCausalLM']}", config_source)

    def test_detection_logits_use_projected_patch_embeddings(self) -> None:
        head_source = read_text("internvl/model/internvl_chat/micro_internvl_head.py")
        model_source = read_text("internvl/model/internvl_chat/modeling_internvl_chat.py")

        self.assertIn("text_dim", head_source)
        self.assertIn("patch_embeddings", head_source)
        self.assertIn("head_out['patch_embeddings']", model_source)
        self.assertNotIn("torch.matmul(patch_norm, text_embeds.T)", model_source)

    def test_detection_loss_supervises_objectness_and_ignores_novel_regions(self) -> None:
        loss_source = read_text("micro_internvl/losses.py")
        train_source = read_text("micro_internvl/train.py")
        dataset_source = read_text("micro_internvl/dataset.py")

        self.assertIn("pred_objectness", loss_source)
        self.assertIn("loss_objectness", loss_source)
        self.assertIn("ignore_boxes", loss_source)
        self.assertIn("outputs[\"pred_objectness\"]", train_source)
        self.assertIn('"ignore_boxes"', dataset_source)

    def test_giou_loss_averages_per_matched_box(self) -> None:
        loss_source = read_text("micro_internvl/losses.py")
        self.assertNotIn("loss_giou = 1 - torch.diag(", loss_source)
        self.assertIn("loss_giou = (1 - torch.diag(", loss_source)

    def test_scheduler_steps_with_optimizer_steps(self) -> None:
        train_source = read_text("micro_internvl/train.py")
        train_one_epoch = train_source[
            train_source.index("def train_one_epoch") : train_source.index("def main")
        ]
        main_loop = train_source[train_source.index("for epoch in range") :]

        self.assertIn("scheduler.step()", train_one_epoch)
        self.assertNotIn("scheduler.step()", main_loop)
        self.assertIn("optimizer_steps_per_epoch", train_source)

    def test_validation_uses_same_amp_dtype_as_training(self) -> None:
        evaluate_source = read_text("micro_internvl/evaluate.py")
        train_source = read_text("micro_internvl/train.py")

        prediction_fn = evaluate_source[
            evaluate_source.index("def micro_internvl_predictions_to_coco") :
            evaluate_source.index("def evaluate_coco")
        ]
        validation_fn = train_source[
            train_source.index("def run_validation") :
            train_source.index("def train_one_epoch")
        ]
        main_loop = train_source[train_source.index("for epoch in range") :]

        self.assertIn("use_amp: bool = False", prediction_fn)
        self.assertIn("amp_dtype: torch.dtype = torch.float32", prediction_fn)
        self.assertIn("torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype)", prediction_fn)
        self.assertRegex(
            prediction_fn,
            r"with torch\.autocast\([^)]*\):\s+outputs = model\(pixel_values=images\)",
        )
        self.assertIn("use_amp: bool", validation_fn)
        self.assertIn("amp_dtype: torch.dtype", validation_fn)
        self.assertIn("use_amp=use_amp", validation_fn)
        self.assertIn("amp_dtype=amp_dtype", validation_fn)
        self.assertIn("use_amp=use_amp", main_loop)
        self.assertIn("amp_dtype=amp_dtype", main_loop)

    def test_nms_casts_reduced_precision_inputs_to_float32(self) -> None:
        utils_source = read_text("micro_internvl/utils.py")
        apply_nms_fn = utils_source[
            utils_source.index("def apply_nms") :
            utils_source.index("def box_cxcywh_to_xyxy")
        ]

        self.assertIn("nms_boxes = boxes.float()", apply_nms_fn)
        self.assertIn("nms_scores = scores.float()", apply_nms_fn)
        self.assertIn("ops.nms(nms_boxes, nms_scores, iou_threshold)", apply_nms_fn)

    def test_prediction_export_casts_reduced_precision_outputs_to_float32(self) -> None:
        evaluate_source = read_text("micro_internvl/evaluate.py")
        prediction_fn = evaluate_source[
            evaluate_source.index("def micro_internvl_predictions_to_coco") :
            evaluate_source.index("def evaluate_coco")
        ]

        self.assertIn("boxes = boxes[keep]", prediction_fn)
        self.assertIn("scores = scores[keep].float()", prediction_fn)
        self.assertIn("boxes_xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0).float()", prediction_fn)
        self.assertIn("boxes_xyxy_abs = boxes_xyxy.cpu().numpy()", prediction_fn)
        self.assertIn("boxes_abs[:, 2] = boxes_xyxy_abs[:, 2] - boxes_xyxy_abs[:, 0]", prediction_fn)
        self.assertIn("boxes_abs[:, 3] = boxes_xyxy_abs[:, 3] - boxes_xyxy_abs[:, 1]", prediction_fn)

    def test_matcher_sanitizes_nonfinite_costs_before_scipy(self) -> None:
        loss_source = read_text("micro_internvl/losses.py")
        matcher_fn = loss_source[
            loss_source.index("class HungarianMatcher") :
            loss_source.index("def sigmoid_focal_loss")
        ]
        detection_loss_fn = loss_source[
            loss_source.index("class DetectionLoss") :
            loss_source.index("class PatchTextAlignmentLoss")
        ]

        self.assertIn("pred_logits = torch.nan_to_num(pred_logits.float()", matcher_fn)
        self.assertIn("pred_boxes = torch.nan_to_num(pred_boxes.float()", matcher_fn)
        self.assertIn("target_boxes = [", matcher_fn)
        self.assertIn("torch.nan_to_num(boxes.float()", matcher_fn)
        self.assertIn("cost = torch.nan_to_num(cost", matcher_fn)
        self.assertIn("posinf=1e6", matcher_fn)
        self.assertIn("neginf=-1e6", matcher_fn)
        self.assertIn("pred_logits = torch.nan_to_num(pred_logits", detection_loss_fn)
        self.assertIn("pred_boxes = torch.nan_to_num(pred_boxes", detection_loss_fn)
        self.assertIn("pred_objectness = torch.nan_to_num(pred_objectness", detection_loss_fn)

    def test_alignment_losses_sanitize_features_before_normalization(self) -> None:
        loss_source = read_text("micro_internvl/losses.py")
        patch_loss_fn = loss_source[
            loss_source.index("class PatchTextAlignmentLoss") :
            loss_source.index("class BoxTextAlignmentLoss")
        ]
        box_loss_fn = loss_source[loss_source.index("class BoxTextAlignmentLoss") :]

        for source in (patch_loss_fn, box_loss_fn):
            self.assertIn("torch.nan_to_num(", source)
            self.assertIn("F.normalize(", source)
            self.assertIn("eps=1e-6", source)
            self.assertIn("logits = torch.nan_to_num(logits", source)

    def test_training_skips_nonfinite_loss_before_backward(self) -> None:
        train_source = read_text("micro_internvl/train.py")
        train_one_epoch = train_source[
            train_source.index("def train_one_epoch") :
            train_source.index("def main")
        ]

        self.assertIn("if not torch.isfinite(loss):", train_one_epoch)
        self.assertIn("Skipping non-finite loss", train_one_epoch)
        self.assertLess(
            train_one_epoch.index("if not torch.isfinite(loss):"),
            train_one_epoch.index("loss = loss / grad_accum_steps"),
        )

    def test_h20_dependency_floor_supports_qwen3(self) -> None:
        requirements = read_text("requirements.txt")
        match = re.search(r"transformers>=([0-9]+)\.([0-9]+)\.([0-9]+)", requirements)
        self.assertIsNotNone(match, "requirements.txt must pin a transformers lower bound")
        major, minor, _patch = map(int, match.groups())
        self.assertGreaterEqual((major, minor), (4, 51))


if __name__ == "__main__":
    unittest.main()
