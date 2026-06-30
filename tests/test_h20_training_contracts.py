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

    def test_h20_dependency_floor_supports_qwen3(self) -> None:
        requirements = read_text("requirements.txt")
        match = re.search(r"transformers>=([0-9]+)\.([0-9]+)\.([0-9]+)", requirements)
        self.assertIsNotNone(match, "requirements.txt must pin a transformers lower bound")
        major, minor, _patch = map(int, match.groups())
        self.assertGreaterEqual((major, minor), (4, 51))


if __name__ == "__main__":
    unittest.main()
