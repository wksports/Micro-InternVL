#!/usr/bin/env python3
"""Tests for the EMDS-7 dataset download helper."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_emds7.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_emds7", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DownloadEmds7Tests(unittest.TestCase):
    def test_selects_named_zip_file_from_figshare_metadata(self) -> None:
        module = load_module()
        files = [
            {"name": "readme.txt", "download_url": "https://example/readme"},
            {"name": "EMDS7.zip", "download_url": "https://example/emds7.zip"},
        ]

        selected = module.select_dataset_file(files, preferred_name="EMDS7.zip")

        self.assertEqual(selected["name"], "EMDS7.zip")
        self.assertEqual(selected["download_url"], "https://example/emds7.zip")

    def test_figshare_metadata_request_uses_browser_headers(self) -> None:
        module = load_module()

        class FakeResponse:
            def read(self):
                return json.dumps({"files": []}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with mock.patch.object(module.urllib.request, "urlopen", return_value=FakeResponse()) as urlopen_mock:
            with self.assertRaisesRegex(RuntimeError, "No files found"):
                module.fetch_figshare_files(16869571)
            request = urlopen_mock.call_args.args[0]
            self.assertIn("Mozilla", request.headers["User-agent"])

    def test_builds_manual_archive_metadata_when_archive_url_is_provided(self) -> None:
        module = load_module()

        selected = module.build_manual_archive_file("https://example.com/EMDS7.zip", "EMDS7.zip")

        self.assertEqual(selected["name"], "EMDS7.zip")
        self.assertEqual(selected["download_url"], "https://example.com/EMDS7.zip")
        self.assertIsNone(selected["size"])

    def test_extracts_zip_and_flattens_images_to_target_directory(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "EMDS7.zip"
            extract_dir = tmp_path / "extract"
            image_dir = tmp_path / "EMDS7"

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("nested/G001/EMDS7-G001-001-0400.png", b"fake")
                zf.writestr("nested/G002/EMDS7-G002-001-0400.jpg", b"fake")
                zf.writestr("nested/readme.txt", b"ignore")

            module.extract_zip(zip_path, extract_dir)
            copied = module.flatten_images(extract_dir, image_dir)

            self.assertEqual(copied, 2)
            self.assertTrue((image_dir / "EMDS7-G001-001-0400.png").exists())
            self.assertTrue((image_dir / "EMDS7-G002-001-0400.jpg").exists())
            self.assertFalse((image_dir / "readme.txt").exists())

    def test_validates_coco_json_image_filenames(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "images"
            annotation_dir = tmp_path / "data"
            image_dir.mkdir()
            annotation_dir.mkdir()
            (image_dir / "present.png").write_bytes(b"fake")
            (annotation_dir / "instances_train.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "present.png"},
                            {"id": 2, "file_name": "missing.png"},
                        ],
                        "annotations": [],
                        "categories": [],
                    }
                ),
                encoding="utf-8",
            )

            missing = module.validate_annotations(annotation_dir, image_dir, splits=("train",))

            self.assertEqual(missing, {"train": ["missing.png"]})


if __name__ == "__main__":
    unittest.main()
