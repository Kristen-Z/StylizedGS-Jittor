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

import os
# import jt
# import torchvision
# import jt.nn.functional as F
from jittor import nn
import jittor as jt
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer_new import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import cv2
import imageio
from PIL import Image
from icecream import ic
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from nnfm_loss import NNFMLoss, match_colors_for_image_set
from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, StyleOptimizationParams
from scene.colmap_loader import qvec2rotmat, rotmat2qvec
from scene.cameras import Camera

def set_geometry_grad(gaussian_model, freeze):
    if freeze:
        # Jittor 中通过 stop_grad() 冻结梯度
        gaussian_model._xyz.stop_grad()
        gaussian_model._scaling.stop_grad()
        gaussian_model._rotation.stop_grad()
        gaussian_model._opacity.stop_grad()
    else:
        # Jittor 中通过 start_grad() 恢复梯度计算
        gaussian_model._xyz.start_grad()
        gaussian_model._scaling.start_grad()
        gaussian_model._rotation.start_grad()
        gaussian_model._opacity.start_grad()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, point_cloud):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    print('image resolution',scene.img_width, scene.img_height)
    nnfm_loss_fn = NNFMLoss(device='cuda')
    if point_cloud:
        xyz, o, s = gaussians.load_ply(point_cloud, reset_basis_dim=1)
        original_xyz, original_opacity, original_scale = jt.array(xyz), jt.array(o), jt.array(s)
        first_iter = 30_000
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = jt.load(checkpoint)
        gaussians.restore(model_params, opt)

    # resize style image
    style_img = imageio.imread(args.style, pilmode="RGB").astype(np.float32) / 255.0 # pilmode="RGB"
    style_h, style_w = style_img.shape[:2]
    content_long_side = max([scene.img_width, scene.img_height])
    if style_h > style_w:
        style_img = cv2.resize(
            style_img,
            (int(content_long_side / style_h * style_w), content_long_side),
            interpolation=cv2.INTER_AREA,
        )
    else:
        style_img = cv2.resize(
            style_img,
            (content_long_side, int(content_long_side / style_w * style_h)),
            interpolation=cv2.INTER_AREA,
        )
    style_img = cv2.resize(
        style_img,
        (style_img.shape[1] // 2, style_img.shape[0] // 2),
        interpolation=cv2.INTER_AREA,
    )
    imageio.imwrite(
        os.path.join(args.model_path, "style_image.jpg"),
        np.clip(style_img * 255.0, 0.0, 255.0).astype(np.uint8),
    )
    style_img = jt.array(style_img)
    # Load style image mask or second style image
    if args.second_style:
        style_img2 = imageio.imread(args.second_style, pilmode="RGB").astype(np.float32) / 255.0
        style_img2 = cv2.resize(style_img2, (style_img.shape[1],style_img.shape[0]), interpolation=cv2.INTER_AREA)
        
        imageio.imwrite(
            os.path.join(args.model_path, "style_image2.jpg"),
            np.clip(style_img2 * 255.0, 0.0, 255.0).astype(np.uint8),
        )
        style_img2 = jt.array(style_img2)


    ic("Style image: ", args.style, style_img.shape)

    saving_iterations.append(opt.iterations)
    testing_iterations.append(opt.iterations)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = jt.array(bg_color, dtype=jt.float32)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    gt_img_list = []
    for view in scene.getTrainCameras():
        if not args.preserve_color:
            gt_img_list.append(view.original_image.permute(1,2,0))
        else:
            gt_img_list.append(view.original_image)

    # prepare depth image & sam mask
    depth_img_list = []
    mask_img_list = []
    mask_half_list = []

    mask_dir = args.mask_dir
    with jt.no_grad():
        for i, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering progress")):
            depth_render = render(view, gaussians, pipe, background)["depth"]
            depth_img_list.append(depth_render)

            select_mask = np.load(os.path.join(args.mask_dir, f'{view.image_name[:-4]}.npy'))
            select_mask = gaussian_filter(select_mask, sigma=1)
            
            mask_img_list.append(jt.array((cv2.resize(select_mask.astype(np.uint8), (scene.img_width, scene.img_height),interpolation=cv2.INTER_AREA)).astype(np.int8)))
            mask_half_list.append(jt.array(cv2.resize(select_mask.astype(np.uint8), (scene.img_width//2, scene.img_height//2),interpolation=cv2.INTER_AREA)))
            

    print("mask",mask_img_list[0].shape, mask_half_list[0].shape)
    # precolor
    if not args.preserve_color:
        gt_imgs = jt.stack(gt_img_list)
        
        if args.second_style:
            gt_imgs1, color_ct = match_colors_for_image_set(gt_imgs, style_img)
            gt_imgs2, color_ct2 = match_colors_for_image_set(gt_imgs, style_img2)

            mask_imgs = jt.stack(mask_img_list).unsqueeze(-1).repeat(1,1,1,3).cuda()

            gt_imgs = gt_imgs1 * (1-mask_imgs) + gt_imgs2 * mask_imgs
            
        else:
            recolor_gt_imgs, color_ct = match_colors_for_image_set(gt_imgs, style_img)
            gt_imgs = recolor_gt_imgs * mask_imgs + gt_imgs * (1-mask_imgs)
        
        # gaussians.apply_ct(color_ct.detach().cpu().numpy())
        gt_img_list = [item.permute(2,0,1) for item in gt_imgs]
        imageio.imwrite(
            os.path.join(args.model_path, "gt_image_recolor.png"),
            np.clip(gt_img_list[0].permute(1,2,0).numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
        )


    for iteration in range(first_iter, opt.iterations + 1):        
        

        gaussians.update_learning_rate(iteration)


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            gt_stack = gt_img_list.copy()
            depth_stack = depth_img_list.copy()
            mask_stack = mask_half_list.copy()
        view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(view_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = jt.rand((3)) if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        
        pred_image, depth_image, _, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg['depth'], render_pkg['mask'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    

        gt_image = gt_stack.pop(view_idx)
        depth_gt = depth_stack.pop(view_idx)
        mask_image = mask_stack.pop(view_idx)

        gt_image = gt_image.unsqueeze(0)
        pred_image = pred_image.unsqueeze(0)

        if iteration > first_iter + 200:
            set_geometry_grad(gaussians,False) # True -> Turn off the geo change
            style_img.stop_grad()
            # style_img2.stop_grad()
            loss_dict = nnfm_loss_fn(
                nn.interpolate(
                    pred_image,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
                style_img.permute(2,0,1).unsqueeze(0),
                blocks=[
                    args.vgg_block,
                ],
                loss_names=["nnfm_loss", "content_loss", "spatial_loss"] if not args.preserve_color else ['lum_nnfm_loss','content_loss', "spatial_loss"],
                contents=nn.interpolate(
                    gt_image,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
                x_mask=mask_image,
                styles2=style_img2.permute(2,0,1).unsqueeze(0) if args.second_style else None,
            )
            loss_dict['nnfm_loss' if not args.preserve_color else 'lum_nnfm_loss'] *= args.style_weight
            loss_dict["content_loss"] *= args.content_weight

            w_variance = jt.mean(jt.pow(pred_image[:, :, :, :-1] - pred_image[:, :, :, 1:], 2))
            h_variance = jt.mean(jt.pow(pred_image[:, :, :-1, :] - pred_image[:, :, 1:, :], 2))
            loss_dict["img_tv_loss"] = opt.img_tv_weight * (h_variance + w_variance) / 2.0
            loss_dict['depth_loss'] = l1_loss(depth_gt, depth_image)
            loss_dict['spatial_loss'] *= args.spatial_weight

        else:
            set_geometry_grad(gaussians,True)
            loss_dict = {}
            Ll1 = l1_loss(pred_image, gt_image)
            loss_dict['ddsm_loss'] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_image, gt_image))
            
            
        loss_dict['opacity_regu'] = l1_loss(gaussians._opacity, original_opacity) * 50
        loss_dict['scale_regu'] = l1_loss(gaussians._scaling, original_scale) * 50
        loss = sum(list(loss_dict.values()))

        gaussians.optimizer.backward(loss)
        # iter_end.record()
        if iteration < opt.densify_until_iter:
            viewspace_point_tensor_grad = gaussians.get_viewspace_point_grad()
        update_flag = False
        with jt.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                def jittor_max(a,b):
                    return jt.where(a>b,a,b)
                if gaussians.max_radii2D.shape == visibility_filter.shape:
                    gaussians.max_radii2D[visibility_filter] = jittor_max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                else:
                    gaussians.max_radii2D = radii[visibility_filter]
                
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    update_flag = True
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
        if iteration < opt.iterations:
                if not update_flag:
                    gaussians.optimizer.step()

        if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                jt.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
   
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = StyleOptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--point_cloud", type=str, default = None)
    parser.add_argument("--gram_iteration", type=int, default = 30_300)
    # ARF params
    parser.add_argument("--style", type=str, help="path to style image")
    parser.add_argument("--style_weight", type=float, default=5, help="style loss weight")
    parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
    parser.add_argument("--spatial_weight", type=float, default=20, help="style loss weight for spatial area")
    parser.add_argument(
        "--vgg_block",
        type=int,
        default=[2, 3],
        help="vgg block for nnfm extracting feature maps",
    )
    parser.add_argument(
        "--reset_basis_dim",
        type=int,
        default=1,
        help="whether to reset the number of spherical harmonics basis to this specified number",
    )
    parser.add_argument("--preserve_color", action="store_true", default=False)
    parser.add_argument('--second_style', type=str, help="path to second style image")
    parser.add_argument("--mask_dir", required=True, help="The directory of multiview masks")
   
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.point_cloud)

    # All done
    print("\nTraining complete.")
