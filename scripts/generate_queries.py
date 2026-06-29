#!/usr/bin/env python3
"""Generate hierarchical queries and hard negatives for Micro-InternVL."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_internvl.dataset import load_category_names
from micro_internvl.queries import HierarchicalQuerySet

logger = logging.getLogger(__name__)

COARSE_TEMPLATES = ["{name}"]
MEDIUM_TEMPLATES = [
    "microscopic image of {name}",
    "{name} microorganism",
    "{name} cell",
]
FINE_TEMPLATES = [
    "{name} under bright-field microscopy",
    "{name} in environmental water sample",
    "{name} observed under optical microscope",
]

DEFAULT_HARD_NEGATIVES = [
    "dust particle",
    "staining artifact",
    "air bubble",
    "debris",
    "background",
    "out-of-focus blur",
    "microscope slide scratch",
    "salt crystal",
]


def generate_cache_queries(category_names: List[str]) -> Dict[str, List[str]]:
    queries = {}
    for name in category_names:
        coarse = [t.format(name=name) for t in COARSE_TEMPLATES]
        medium = [t.format(name=name) for t in MEDIUM_TEMPLATES]
        fine = [t.format(name=name) for t in FINE_TEMPLATES]
        seen = set()
        all_queries = []
        for q in coarse + medium + fine:
            if q not in seen:
                seen.add(q)
                all_queries.append(q)
        queries[name] = all_queries
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate hierarchical queries for Micro-InternVL")
    parser.add_argument("--config", type=str, default="micro_internvl/config.yaml", help="Path to config YAML")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config_path.parent
    category_map_path = (base_dir / config["data"]["category_map"]).resolve()
    _, _, idx_to_name = load_category_names(str(category_map_path))
    category_names = [idx_to_name[i] for i in range(len(idx_to_name))]

    logger.info(f"Generating cache queries for {len(category_names)} categories")
    queries = generate_cache_queries(category_names)
    query_set = HierarchicalQuerySet(queries, DEFAULT_HARD_NEGATIVES)

    if args.out is None:
        out_path = base_dir / "data" / "emds7" / "queries.json"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    query_set.to_file(str(out_path))
    logger.info(f"Saved queries to {out_path}")


if __name__ == "__main__":
    main()
