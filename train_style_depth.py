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
import cv2
import os
import jittor as jt
from jittor import nn
# import torchvision
# import jt.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer_new import render
import sys
from scene import Scene, GaussianModel
import uuid
import cv2
import imageio
from icecream import ic
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from nnfm_loss import NNFMLoss, match_colors_for_image_set, color_histgram_match
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, StyleOptimizationParams
TENSORBOARD_FOUND = False

def set_geometry_grad(gaussian_model, freeze):
    if freeze:
        gaussian_model._xyz.stop_grad()
        gaussian_model._scaling.stop_grad()
        gaussian_model._rotation.stop_grad()
        gaussian_model._opacity.stop_grad()
    else:
        gaussian_model._xyz.start_grad()
        gaussian_model._scaling.start_grad()
        gaussian_model._rotation.start_grad()
        gaussian_model._opacity.start_grad()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, point_cloud):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
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
    style_img_copy = jt.array(style_img)
    ic("Style image: ", args.style, style_img.shape)

    saving_iterations.append(opt.iterations)
    testing_iterations.append(opt.iterations)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = jt.array(bg_color, dtype=jt.float32)


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    os.makedirs(os.path.join(args.model_path,'inter_res'),exist_ok=True) # debug use

    if args.second_style:
        style_img2 = imageio.imread(args.second_style, pilmode="RGB").astype(np.float32) / 255.0
        style_img2 = jt.array(style_img2)
    if not args.preserve_color:
        gt_img_list = []
        for view in scene.getTrainCameras():
            gt_img_list.append(view.original_image.permute(1,2,0))
        gt_imgs = jt.stack(gt_img_list)   

        if args.histgram_match:
            ic(gt_imgs.shape)
            gt_imgs, color_ct = color_histgram_match(gt_imgs, style_img_copy if not args.second_style else style_img2) #.repeat(gt_imgs.shape[0],1,1,1))
        else:
            gt_imgs, color_ct = match_colors_for_image_set(gt_imgs, style_img_copy if not args.second_style else style_img2)
        gaussians.apply_ct(color_ct.numpy())
        gt_img_list = [item.permute(2,0,1) for item in gt_imgs]
        imageio.imwrite(
            os.path.join(args.model_path, "gt_image_recolor.png"),
            np.clip(gt_img_list[0].permute(1,2,0).numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
        )

    # scale control
    scale_coef = []
    if args.scale_level is not None:
        # scale_coefs=[
        #             [8,0,0,0,0,0], #0
        #             [4,4,0,0,0,0], #1
        #             [0,0,5,3,0,0], #2
        #             [0,0,0,5,3,0], #3
        #             [0,0,2,2,2,2], #4
        #             [0,0,0,0,0,8], #5
        #         ]

        scale_coefs=[
                    [0,0,0,0,4,4], #0
                    [0,0,5,3,0,0], #1
                    [8,0,0,0,0,0], #2
                ]
        print("scale level:",args.scale_level, scale_coefs[args.scale_level])
        assert(args.scale_level < len(scale_coefs))
        scale_coef = scale_coefs[args.scale_level]

    # prepare depth image
    depth_img_list = []
    with jt.no_grad():
        for view in tqdm(scene.getTrainCameras(), desc="Rendering progress"):
                rgb = render(view, gaussians, pipe, background)["render"]
                depth_render = render(view, gaussians, pipe, background)["depth"]
                depth_img_list.append(depth_render)

    for iteration in range(first_iter, opt.iterations + 1):        

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not args.preserve_color:
                gt_stack = gt_img_list.copy()
            depth_stack = depth_img_list.copy()
        view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(view_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = jt.rand((3)) if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        pred_image, depth_image, mask_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg['depth'], render_pkg['mask'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
        if not args.preserve_color:
            gt_image = gt_stack.pop(view_idx)
        else:
            gt_image = viewpoint_cam.original_image
        depth_gt = depth_stack.pop(view_idx)

        gt_image = gt_image.unsqueeze(0)
        pred_image = pred_image.unsqueeze(0)

        if args.preserve_color or args.second_style:
            loss_type = ['lum_nnfm_loss','content_loss']
        elif args.scale_level is not None:
            loss_type = ["scale_loss", "content_loss"]
        else:
            loss_type = ['nnfm_loss','content_loss']

        if iteration >= first_iter+200:
            set_geometry_grad(gaussians,False)
            style_img = style_img_copy.copy()
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
                loss_names=loss_type,
                contents=nn.interpolate(
                    gt_image,
                    size=None,
                    scale_factor=0.5,
                    mode="bilinear",
                ),
                layer_coef=scale_coef
            )
            w_variance = jt.mean(jt.pow(pred_image[:, :, :, :-1] - pred_image[:, :, :, 1:], 2))
            h_variance = jt.mean(jt.pow(pred_image[:, :, :-1, :] - pred_image[:, :, 1:, :], 2))

            loss_dict[loss_type[0]] *= args.style_weight
            loss_dict["content_loss"] *= args.content_weight
            loss_dict["img_tv_loss"] = opt.img_tv_weight * (h_variance + w_variance) / 2.0
            loss_dict['depth_loss'] = l2_loss(depth_image, depth_gt)
            loss_dict['color_loss'] = l2_loss(pred_image, gt_image)
            
        else:
            set_geometry_grad(gaussians,True)
            
            loss_dict = {}
            Ll1 = l1_loss(pred_image, gt_image)
            loss_dict['ddsm_loss'] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_image, gt_image))
            

        # opacity & scale regulariers
        loss_dict['opacity_regu'] = l1_loss(gaussians._opacity, original_opacity)
        loss_dict['scale_regu'] = l1_loss(gaussians._scaling, original_scale)
        
        loss = sum(list(loss_dict.values()))
        gaussians.optimizer.backward(loss)
        
        
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


def training_report(tb_writer, iteration, loss_dict, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        for x in loss_dict:
            tb_writer.add_scalar(f'train_loss_patches/{x}', loss_dict[x].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        jt.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = jt.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = jt.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        jt.cuda.empty_cache()

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
    # Style params
    parser.add_argument("--style", type=str, help="path to style image")
    parser.add_argument("--second_style", type=str, default="", help="path to second style image")
    parser.add_argument("--histgram_match", action="store_true", default=True)
    parser.add_argument("--style_weight", type=float, default=5, help="style loss weight")
    parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
    parser.add_argument(
        "--vgg_block",
        type=int,
        default=2,
        help="vgg block for nnfm extracting feature maps",
    )
    parser.add_argument(
        "--reset_basis_dim",
        type=int,
        default=1,
        help="whether to reset the number of spherical harmonics basis to this specified number",
    )
    parser.add_argument("--preserve_color", action="store_true", default=False)
    parser.add_argument("--histgram_match", action="store_true", default=False)
    parser.add_argument("--scale_level", type=int, default =None, choices=[0,1,2], help='the scale of style pattern, can be [0,1,2]')

   
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.point_cloud)

    # All done
    print("\nTraining complete.")
