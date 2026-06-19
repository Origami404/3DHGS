#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import loadCam
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov


def camera_center_from_colmap(qvec, tvec):
    rotation_world_to_camera = qvec2rotmat(qvec)
    return -rotation_world_to_camera.T @ np.asarray(tvec)


def forward_from_colmap(qvec):
    rotation_world_to_camera = qvec2rotmat(qvec)
    return rotation_world_to_camera.T @ np.array([0.0, 0.0, 1.0])


def rotate_camera_yaw(camera, degrees):
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = camera.R.T
    world_to_camera[:3, 3] = camera.T

    camera_to_world = np.linalg.inv(world_to_camera)
    angle = math.radians(degrees)
    yaw_camera = np.array([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ], dtype=np.float64)
    camera_to_world[:3, :3] = camera_to_world[:3, :3] @ yaw_camera

    new_world_to_camera = np.linalg.inv(camera_to_world)
    new_r = new_world_to_camera[:3, :3].T
    new_t = new_world_to_camera[:3, 3]

    return Camera(
        colmap_id=camera.colmap_id,
        R=new_r,
        T=new_t,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=camera.original_image.detach().cpu(),
        gt_alpha_mask=None,
        image_name=f"{camera.image_name}_yaw_{degrees:+g}",
        uid=camera.uid,
        trans=camera.trans,
        scale=camera.scale,
        data_device="cuda",
    )


def tensor_to_pil(image_tensor):
    image_tensor = image_tensor.detach().clamp(0.0, 1.0).cpu()
    array = (image_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def fit_panel(image, size, title):
    panel_width, panel_height = size
    title_height = 34
    image_area_height = panel_height - title_height

    image = image.convert("RGB")
    image.thumbnail((panel_width, image_area_height), Image.LANCZOS)

    panel = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_width, title_height), fill=(30, 30, 30))
    draw.text((10, 9), title, fill=(255, 255, 255))
    panel.paste(image, ((panel_width - image.width) // 2, title_height + (image_area_height - image.height) // 2))
    return panel


def make_sheet(original, rendered_same, rendered_left, rendered_right, info_lines, output_path):
    panel_size = (480, 300)
    cols = 2
    rows = 2
    text_height = 150

    panels = [
        fit_panel(original, panel_size, "original image"),
        fit_panel(rendered_same, panel_size, "render: same COLMAP pose"),
        fit_panel(rendered_left, panel_size, "render: yaw -8 deg"),
        fit_panel(rendered_right, panel_size, "render: yaw +8 deg"),
    ]

    sheet = Image.new("RGB", (cols * panel_size[0], rows * panel_size[1] + text_height), (255, 255, 255))
    for index, panel in enumerate(panels):
        x = (index % cols) * panel_size[0]
        y = (index // cols) * panel_size[1]
        sheet.paste(panel, (x, y))

    draw = ImageDraw.Draw(sheet)
    y0 = rows * panel_size[1] + 12
    for line in info_lines:
        draw.text((18, y0), line, fill=(0, 0, 0))
        y0 += 22

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Render pose-aware preview sheets for 3D-HGS scenes.")
    parser.add_argument("--source", "-s", default="t2data/Train_colmap")
    parser.add_argument("--model", "-m", default="output/Train_3dhgs")
    parser.add_argument("--image", default="00001.jpg")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--yaw", type=float, default=8.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    safe_state(True)

    source_path = Path(args.source)
    model_path = Path(args.model)
    image_name = Path(args.image).name
    image_stem = Path(image_name).stem
    output_path = Path(args.output) if args.output else model_path / "previews" / f"pose_render_preview_{image_stem}.jpg"

    scene_info = readColmapSceneInfo(str(source_path), images=None, eval=False)
    cam_info_by_name = {Path(info.image_path).name: info for info in scene_info.train_cameras}
    if image_name not in cam_info_by_name:
        raise ValueError(f"Image {image_name!r} was not found in COLMAP cameras under {source_path}")
    cam_info = cam_info_by_name[image_name]

    dataset_args = SimpleNamespace(
        resolution=-1,
        data_device="cuda",
    )
    camera = loadCam(dataset_args, 0, cam_info, 1.0)

    gaussians = GaussianModel(3)
    point_cloud_path = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not point_cloud_path.exists():
        raise FileNotFoundError(f"Cannot find trained point cloud: {point_cloud_path}")
    gaussians.load_ply(str(point_cloud_path))

    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        same = render(camera, gaussians, pipeline, background)["render"]
        left_camera = rotate_camera_yaw(camera, -args.yaw)
        right_camera = rotate_camera_yaw(camera, args.yaw)
        left = render(left_camera, gaussians, pipeline, background)["render"]
        right = render(right_camera, gaussians, pipeline, background)["render"]

    extrinsics = read_extrinsics_binary(str(source_path / "sparse" / "0" / "images.bin"))
    by_name = {record.name: record for record in extrinsics.values()}
    record = by_name[image_name]
    camera_center = camera_center_from_colmap(record.qvec, record.tvec)
    forward = forward_from_colmap(record.qvec)

    info_lines = [
        image_name,
        f"image_id: {record.id}",
        f"camera_id: {record.camera_id}",
        "camera_center_world: " + np.array2string(camera_center, precision=6, suppress_small=False),
        "forward_world: " + np.array2string(forward, precision=6, suppress_small=False),
    ]

    make_sheet(
        original=tensor_to_pil(camera.original_image[:3]),
        rendered_same=tensor_to_pil(same),
        rendered_left=tensor_to_pil(left),
        rendered_right=tensor_to_pil(right),
        info_lines=info_lines,
        output_path=output_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
