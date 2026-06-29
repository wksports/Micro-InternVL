# --------------------------------------------------------
# Micro-InternVL detection head
# Added on top of official InternVLChatModel
# --------------------------------------------------------

from typing import Dict

import torch
import torch.nn as nn


class MicroInternVLDetectionHead(nn.Module):
    """Lightweight detection head operating on uncompressed InternViT patch tokens.

    For InternVL3.5 at 448x448 resolution, the vision encoder outputs 1024 patch
    tokens (32x32 grid) before pixel-shuffle downsampling. This head predicts a
    bounding box and objectness score for each patch. Open-vocabulary class scores
    are computed externally as similarity between patch features and text embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act)
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act,
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self.objectness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act,
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, patch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_features: [B, 1024, D]

        Returns:
            Dict with pred_boxes [B, 1024, 4] (cx, cy, w, h normalized)
            and pred_objectness [B, 1024, 1].
        """
        features = self.trunk(patch_features)
        pred_boxes = self.box_head(features)
        pred_objectness = self.objectness_head(features)
        return {
            "pred_boxes": pred_boxes,
            "pred_objectness": pred_objectness,
        }
