"""Micro-InternVL model wrapper around official InternVLChatModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from internvl.model.internvl_chat import InternVLChatModel


class MicroInternVL(nn.Module):
    """Thin wrapper that loads official InternVLChatModel with Micro-InternVL config."""

    def __init__(
        self,
        model_path: str,
        micro_internvl_config: Dict[str, Any],
        torch_dtype: str = "bfloat16",
        device_map: Optional[str] = None,
    ):
        super().__init__()
        self.micro_internvl_config = micro_internvl_config

        dtype = self._parse_dtype(torch_dtype)

        # The official InternVL3.5-4B GitHub-format checkpoint is loaded via
        # InternVLChatModel.from_pretrained. We inject micro_internvl_config into
        # the model config by passing it as a keyword argument.
        self.base_model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            micro_internvl_config=micro_internvl_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.base_model.set_micro_internvl_tokenizer(self.tokenizer)

    @staticmethod
    def _parse_dtype(dtype_str: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype_str.lower() not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {dtype_str}")
        return mapping[dtype_str.lower()]

    def encode_text_queries(self, queries: List[str]) -> torch.Tensor:
        return self.base_model.encode_text_queries(queries)

    def set_text_embeddings(self, text_embeds: torch.Tensor) -> None:
        self.base_model.set_text_embeddings(text_embeds)

    @property
    def text_embeddings(self) -> Optional[torch.Tensor]:
        return self.base_model.text_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_queries: Optional[List[str]] = None,
        return_patch_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if text_queries is not None:
            text_embeds = self.encode_text_queries(text_queries)
        else:
            text_embeds = self.text_embeddings

        return self.base_model.forward_detection(
            pixel_values=pixel_values,
            text_embeddings=text_embeds,
            return_patch_features=return_patch_features,
        )

    def save_pretrained(self, save_directory: str) -> None:
        """Save base model, detection head, tokenizer, and Micro-InternVL config."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        import yaml
        with open(save_path / "micro_internvl_config.yaml", "w") as f:
            yaml.dump(self.micro_internvl_config, f, default_flow_style=False)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        micro_internvl_config: Optional[Dict[str, Any]] = None,
        torch_dtype: str = "bfloat16",
        device_map: Optional[str] = None,
    ) -> "MicroInternVL":
        load_path = Path(load_directory)

        import yaml
        if micro_internvl_config is None:
            with open(load_path / "micro_internvl_config.yaml", "r") as f:
                micro_internvl_config = yaml.safe_load(f)

        instance = cls(
            model_path=str(load_path),
            micro_internvl_config=micro_internvl_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        return instance

    def train(self, mode: bool = True):
        # Keep the language model frozen regardless of train mode
        self.base_model.train(mode)
        if hasattr(self.base_model, "language_model"):
            self.base_model.language_model.eval()
        return self

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def parameters(self):
        # Exclude frozen language model parameters from optimizer
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                yield param

    def named_parameters(self):
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                yield name, param
