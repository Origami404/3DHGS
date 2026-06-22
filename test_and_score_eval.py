#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
loss_fn_alex = lpips.LPIPS(net='vgg')
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def empty_scores(device):
    return {
        "count": 0,
        "psnr": torch.tensor(0.0, dtype=torch.float64, device=device),
        "ssim": torch.tensor(0.0, dtype=torch.float64, device=device),
        "lpips": torch.tensor(0.0, dtype=torch.float64, device=device),
    }


def add_scores(total, scores):
    total["count"] += scores["count"]
    total["psnr"] += scores["psnr"]
    total["ssim"] += scores["ssim"]
    total["lpips"] += scores["lpips"]


def print_scores(name, scores):
    if scores["count"] == 0:
        print(f"{name}: no views")
        return

    print(f"{name} views=", scores["count"])
    print(f"{name} PSNR=", scores["psnr"] / scores["count"])
    print(f"{name} SSIM=", scores["ssim"] / scores["count"])
    print(f"{name} LPIPS=", scores["lpips"] / scores["count"])


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)

    scores = empty_scores(gaussians.get_features.device)

    loss_fn_alex.to(gaussians.get_features.device)

    for view in tqdm(views, desc="Rendering {} progress".format(name)):
        image = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]

        scores["count"] += 1
        scores["psnr"] += psnr(image, gt).mean().double()
        scores["ssim"] += ssim(image, gt).double()
        img1 = image.unsqueeze(0)
        img2 = gt.unsqueeze(0)
        scores["lpips"] += loss_fn_alex.forward(img1, img2).squeeze().double()

        filename = view.image_name + ".png"
        torchvision.utils.save_image(image, os.path.join(render_path, filename))
        torchvision.utils.save_image(gt, os.path.join(gts_path, filename))

    return scores

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, ignore_points=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        total_scores = empty_scores(background.device)

        if not skip_train:
            train_scores = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            print_scores("Train", train_scores)
            add_scores(total_scores, train_scores)

        if not skip_test:
            test_scores = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            print_scores("Test", test_scores)
            add_scores(total_scores, test_scores)

        print_scores("Combined", total_scores)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
