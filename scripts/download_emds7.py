#!/usr/bin/env python3
"""Download and prepare the EMDS-7 image dataset.

The repository already contains COCO-format annotations under data/emds7.
This helper downloads the public EMDS-7 image archive from Figshare, extracts
it, and gathers images into raw/emds7/EMDS7 so the training config can use them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm


DEFAULT_ARTICLE_ID = 16869571
DEFAULT_FILE_NAME = "EMDS7.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def fetch_figshare_files(article_id: int) -> List[Dict]:
    """Fetch file metadata for a Figshare article."""
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch Figshare metadata from {url}: {exc}") from exc

    files = payload.get("files", [])
    if not files:
        raise RuntimeError(f"No files found in Figshare article {article_id}")
    return files


def select_dataset_file(files: Sequence[Dict], preferred_name: str = DEFAULT_FILE_NAME) -> Dict:
    """Select the dataset zip from Figshare file metadata."""
    for item in files:
        if item.get("name") == preferred_name:
            return item

    zip_files = [item for item in files if str(item.get("name", "")).lower().endswith(".zip")]
    if len(zip_files) == 1:
        return zip_files[0]
    if not zip_files:
        names = ", ".join(str(item.get("name")) for item in files)
        raise RuntimeError(f"No zip file found in Figshare files: {names}")

    names = ", ".join(str(item.get("name")) for item in zip_files)
    raise RuntimeError(f"Multiple zip files found ({names}); pass --file-name to choose one")


def download_file(url: str, destination: Path, expected_size: int | None = None, force: bool = False) -> None:
    """Download a URL to destination with a byte progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        current_size = destination.stat().st_size
        if expected_size is None or current_size == expected_size:
            print(f"Using existing archive: {destination}")
            return
        print(f"Existing archive size mismatch ({current_size} != {expected_size}); re-downloading.")

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "Micro-InternVL dataset downloader"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = expected_size or int(response.headers.get("Content-Length") or 0) or None
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {destination.name}",
            ) as progress:
                with open(tmp_path, "wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        progress.update(len(chunk))
    except urllib.error.URLError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    tmp_path.replace(destination)


def extract_zip(zip_path: Path, extract_dir: Path, force: bool = False) -> None:
    """Extract a zip archive to a directory."""
    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        print(f"Using existing extracted directory: {extract_dir}")
        return

    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}", unit="file"):
            archive.extract(member, extract_dir)


def iter_image_files(source_dir: Path) -> Iterable[Path]:
    """Yield image files recursively from source_dir."""
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def flatten_images(source_dir: Path, image_dir: Path, force: bool = False) -> int:
    """Copy all images from source_dir recursively into image_dir."""
    image_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    image_files = list(iter_image_files(source_dir))
    for src in tqdm(image_files, desc=f"Preparing {image_dir.name}", unit="image"):
        dst = image_dir / src.name
        if dst.exists() and not force:
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def validate_annotations(
    annotation_dir: Path,
    image_dir: Path,
    splits: Sequence[str] = ("train", "val", "test"),
) -> Dict[str, List[str]]:
    """Return missing image filenames by split for existing COCO annotation files."""
    missing_by_split: Dict[str, List[str]] = {}
    for split in splits:
        annotation_file = annotation_dir / f"instances_{split}.json"
        if not annotation_file.exists():
            continue
        with open(annotation_file, "r", encoding="utf-8") as handle:
            coco = json.load(handle)
        missing = []
        for image in coco.get("images", []):
            filename = image.get("file_name")
            if filename and not (image_dir / filename).exists():
                missing.append(filename)
        missing_by_split[split] = missing
    return missing_by_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare EMDS-7 images for Micro-InternVL")
    parser.add_argument("--article-id", type=int, default=DEFAULT_ARTICLE_ID, help="Figshare article id")
    parser.add_argument("--file-name", type=str, default=DEFAULT_FILE_NAME, help="Dataset zip file name on Figshare")
    parser.add_argument("--output-dir", type=Path, default=Path("raw/emds7"), help="Directory for archive and extraction")
    parser.add_argument("--image-dir", type=Path, default=Path("raw/emds7/EMDS7"), help="Final flat image directory")
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/emds7"), help="COCO annotation directory")
    parser.add_argument("--force", action="store_true", help="Re-download, re-extract, and overwrite copied images")
    parser.add_argument("--skip-extract", action="store_true", help="Only download the archive")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    image_dir = args.image_dir.resolve()
    annotation_dir = args.annotation_dir.resolve()

    print(f"Fetching Figshare metadata for article {args.article_id}...")
    files = fetch_figshare_files(args.article_id)
    selected = select_dataset_file(files, preferred_name=args.file_name)

    archive_name = selected["name"]
    download_url = selected["download_url"]
    expected_size = selected.get("size")
    archive_path = output_dir / archive_name
    extract_dir = output_dir / "extracted"

    print(f"Selected file: {archive_name}")
    print(f"Download URL: {download_url}")
    download_file(download_url, archive_path, expected_size=expected_size, force=args.force)

    if args.skip_extract:
        print("Skipping extraction because --skip-extract was set.")
        return 0

    extract_zip(archive_path, extract_dir, force=args.force)
    copied = flatten_images(extract_dir, image_dir, force=args.force)
    print(f"Copied {copied} images into {image_dir}")

    missing_by_split = validate_annotations(annotation_dir, image_dir)
    total_missing = sum(len(items) for items in missing_by_split.values())
    for split, missing in missing_by_split.items():
        print(f"{split}: missing {len(missing)} images")
        if missing:
            print("  examples:", ", ".join(missing[:5]))

    if total_missing:
        print(
            f"Found {total_missing} missing image references. "
            "Check that the downloaded dataset version matches data/emds7 annotations.",
            file=sys.stderr,
        )
        return 2

    print("Dataset is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
