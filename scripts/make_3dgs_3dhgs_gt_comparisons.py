#!/usr/bin/env python3
"""Create side-by-side 3D-GS / 3D-HGS / Ground-Truth comparison images.

Default paths match docs/复现日志.md for Tanks & Temples train test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text
from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text


LABELS = ("3D-GS", "3D-HGS", "Ground-Truth")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate horizontal 3D-GS / 3D-HGS / GT comparison images with COLMAP camera metadata."
    )
    parser.add_argument(
        "--gs-dir",
        type=Path,
        default=Path("data/3dgs_test_ours_30000/train/test/ours_30000/renders"),
        help="Directory containing 3D-GS rendered images.",
    )
    parser.add_argument(
        "--hgs-dir",
        type=Path,
        default=Path("data/output/tandt/train/test/ours_30000/renders"),
        help="Directory containing 3D-HGS rendered images.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data/output/tandt/train/test/ours_30000/gt"),
        help="Directory containing ground-truth images.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=Path("data/input/tandt/train/sparse/0"),
        help="COLMAP sparse/0 directory containing images.bin/cameras.bin or images.txt/cameras.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/comparisons/tandt/train/test_ours_30000"),
        help="Directory to write comparison images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of comparison images to generate.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Specific image stems or filenames to generate, e.g. 00001 00009.png.",
    )
    parser.add_argument(
        "--text-height",
        type=int,
        default=180,
        help="Canvas height reserved for camera metadata text.",
    )
    parser.add_argument(
        "--label-height",
        type=int,
        default=34,
        help="Canvas height reserved for top labels.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=16,
        help="Padding around labels and metadata text.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=18,
        help="Font size for labels and metadata.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist instead of overwriting them.",
    )
    return parser.parse_args()


def normalize_stem(name: str) -> str:
    return Path(name).stem


def list_stems(directory: Path) -> set[str]:
    stems = set()
    for extension in IMAGE_EXTENSIONS:
        stems.update(path.stem for path in directory.glob(f"*{extension}"))
        stems.update(path.stem for path in directory.glob(f"*{extension.upper()}"))
    return stems


def find_image(directory: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        for candidate in (directory / f"{stem}{extension}", directory / f"{stem}{extension.upper()}"):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No image found for stem {stem!r} in {directory}")


def load_colmap_metadata(sparse_dir: Path):
    images_bin = sparse_dir / "images.bin"
    cameras_bin = sparse_dir / "cameras.bin"
    images_txt = sparse_dir / "images.txt"
    cameras_txt = sparse_dir / "cameras.txt"

    if images_bin.exists() and cameras_bin.exists():
        images = read_extrinsics_binary(str(images_bin))
        cameras = read_intrinsics_binary(str(cameras_bin))
    elif images_txt.exists() and cameras_txt.exists():
        images = read_extrinsics_text(str(images_txt))
        cameras = read_intrinsics_text(str(cameras_txt))
    else:
        raise FileNotFoundError(
            f"Could not find COLMAP images/cameras files under {sparse_dir}. "
            "Expected images.bin+cameras.bin or images.txt+cameras.txt."
        )

    return {Path(image.name).stem: image for image in images.values()}, cameras


def format_array(values: Iterable[float], precision: int = 6) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def format_matrix(matrix: np.ndarray, precision: int = 6) -> list[str]:
    return ["[" + ", ".join(f"{float(value):.{precision}f}" for value in row) + "]" for row in matrix]


def camera_center_from_colmap(image) -> np.ndarray:
    rotation = image.qvec2rotmat()
    return -rotation.T @ image.tvec


def metadata_lines(stem: str, images_by_stem: dict, cameras: dict) -> list[str]:
    image = images_by_stem.get(stem)
    if image is None:
        return [f"image_name: {stem}", "COLMAP metadata: not found"]

    camera = cameras[image.camera_id]
    center = camera_center_from_colmap(image)
    rotation = image.qvec2rotmat()
    rotation_lines = format_matrix(rotation)

    return [
        f"image_name: {image.name}    image_id: {image.id}    camera_id: {image.camera_id}",
        f"camera_model: {camera.model}    size: {camera.width}x{camera.height}    params: {format_array(camera.params)}",
        f"qvec (world->camera): {format_array(image.qvec)}",
        f"tvec (world->camera): {format_array(image.tvec)}",
        f"camera_center_world: {format_array(center)}",
        "R world->camera:",
        f"  {rotation_lines[0]}",
        f"  {rotation_lines[1]}",
        f"  {rotation_lines[2]}",
    ]


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill):
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = left + (right - left - text_width) // 2
    y = top + (bottom - top - text_height) // 2
    draw.text((x, y), text, font=font, fill=fill)


def paste_fit(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int], fill=(0, 0, 0)):
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    image = image.convert("RGB")
    scale = min(box_width / image.width, box_height / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", (box_width, box_height), fill)
    paste_xy = ((box_width - resized.width) // 2, (box_height - resized.height) // 2)
    background.paste(resized, paste_xy)
    canvas.paste(background, (left, top))


def make_comparison(stem: str, args: argparse.Namespace, images_by_stem: dict, cameras: dict, font, title_font):
    paths = [
        find_image(args.gs_dir, stem),
        find_image(args.hgs_dir, stem),
        find_image(args.gt_dir, stem),
    ]
    images = [Image.open(path).convert("RGB") for path in paths]
    panel_width = max(image.width for image in images)
    panel_height = max(image.height for image in images)
    width = panel_width * 3
    height = args.label_height + panel_height + args.text_height

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, (label, image) in enumerate(zip(LABELS, images)):
        left = idx * panel_width
        label_box = (left, 0, left + panel_width, args.label_height)
        draw.rectangle(label_box, fill=(28, 28, 28))
        draw_centered_text(draw, label_box, label, title_font, fill=(255, 255, 255))
        paste_fit(canvas, image, (left, args.label_height, left + panel_width, args.label_height + panel_height))
        if idx > 0:
            draw.line((left, 0, left, args.label_height + panel_height), fill=(255, 255, 255), width=2)

    text_top = args.label_height + panel_height
    draw.rectangle((0, text_top, width, height), fill=(245, 245, 245))
    lines = metadata_lines(stem, images_by_stem, cameras)
    line_height = max(font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 4, args.font_size + 4)
    y = text_top + args.padding
    x = args.padding
    for line in lines:
        if y + line_height > height - args.padding:
            break
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += line_height

    for image in images:
        image.close()

    output_path = args.output_dir / f"{stem}.png"
    if output_path.exists() and args.skip_existing:
        return output_path, False
    canvas.save(output_path)
    return output_path, True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for directory in (args.gs_dir, args.hgs_dir, args.gt_dir, args.sparse_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Required path does not exist: {directory}")

    if args.names:
        stems = [normalize_stem(name) for name in args.names]
    else:
        stems = sorted(list_stems(args.gs_dir) & list_stems(args.hgs_dir) & list_stems(args.gt_dir))

    if args.limit is not None:
        stems = stems[: args.limit]
    if not stems:
        raise RuntimeError("No common image names found to compare.")

    images_by_stem, cameras = load_colmap_metadata(args.sparse_dir)
    font = load_font(args.font_size)
    title_font = load_font(max(args.font_size + 2, 20))

    written = 0
    skipped = 0
    for stem in stems:
        output_path, did_write = make_comparison(stem, args, images_by_stem, cameras, font, title_font)
        if did_write:
            written += 1
            print(f"wrote {output_path}")
        else:
            skipped += 1
            print(f"skip existing {output_path}")

    print(f"done: written={written}, skipped={skipped}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
