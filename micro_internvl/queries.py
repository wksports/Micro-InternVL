"""Hierarchical query management for Micro-InternVL.

Supports loading pre-generated queries and hard negatives, sampling query
levels during training, and ensembling at inference.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


class HierarchicalQuerySet:
    """Container for hierarchical queries and hard negatives."""

    def __init__(self, query_dict: Dict[str, List[str]], hard_negatives: List[str]):
        """Initialize from a query dictionary.

        Args:
            query_dict: {category_name: [coarse, medium, fine, ...]}
            hard_negatives: List of hard negative text descriptions.
        """
        self.query_dict = query_dict
        self.hard_negatives = hard_negatives

    def get_queries(self, category_names: List[str], level: Optional[str] = None) -> List[str]:
        """Get queries for a list of categories.

        Args:
            category_names: List of category names.
            level: One of 'coarse', 'medium', 'fine', or None. If None,
                returns the finest available query for each category.

        Returns:
            List of query strings aligned with category_names.
        """
        level_map = {"coarse": 0, "medium": 1, "fine": 2}
        queries = []
        for name in category_names:
            if name not in self.query_dict or len(self.query_dict[name]) == 0:
                queries.append(name)
                continue

            if level is None:
                queries.append(self.query_dict[name][-1])
            elif level in level_map:
                idx = min(level_map[level], len(self.query_dict[name]) - 1)
                queries.append(self.query_dict[name][idx])
            else:
                queries.append(self.query_dict[name][-1])
        return queries

    def sample_queries(self, category_names: List[str]) -> List[str]:
        """Sample one query level per category (for training)."""
        queries = []
        for name in category_names:
            if name not in self.query_dict or len(self.query_dict[name]) == 0:
                queries.append(name)
            else:
                queries.append(random.choice(self.query_dict[name]))
        return queries

    def ensemble_queries(self, category_names: List[str]) -> List[List[str]]:
        """Return all available query levels per category (for inference ensemble)."""
        return [self.query_dict.get(name, [name]) for name in category_names]

    def get_hard_negatives(self) -> List[str]:
        return self.hard_negatives

    @classmethod
    def from_file(cls, path: str) -> "HierarchicalQuerySet":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        query_dict = data.get("queries", {})
        hard_negatives = data.get("hard_negatives", [])
        return cls(query_dict, hard_negatives)

    def to_file(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"queries": self.query_dict, "hard_negatives": self.hard_negatives},
                f,
                indent=2,
                ensure_ascii=False,
            )
