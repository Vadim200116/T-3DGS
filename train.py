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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, anisotropic_total_variation_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, make_video, prep_img, mask_frame
from natsort import natsorted
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.transient_utils import LinearSegmentationHead, DinoFeatureExatractor, dilate_mask
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

ENABLE_TRANSIENT = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, fps):
    global ENABLE_TRANSIENT

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    masks_bank = {}
    if not pipe.disable_transient:
        feature_extractor = DinoFeatureExatractor(pipe.dino_version)
        transient_model = LinearSegmentationHead(1, feature_extractor.dino_model.embed_dim).cuda()
        transient_model.train()
        transient_optimizer = torch.optim.Adam(transient_model.parameters(), lr=1e-3)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_tr_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        diff = l1_loss(image, gt_image, average=False)
        ssim_value = (ssim(image, gt_image, size_average=False))

        transient_loss = 0

        if not pipe.disable_transient:
            if iteration == opt.transient_from_iter:
                ENABLE_TRANSIENT = True
            elif iteration > opt.transient_until_iter:
                ENABLE_TRANSIENT = False
            elif iteration < opt.densify_until_iter and iteration % opt.opacity_reset_interval == 0:
                ENABLE_TRANSIENT = False
            elif iteration > opt.transient_from_iter and (iteration - opt.transient_buffer_interval) % opt.opacity_reset_interval == 0:
                ENABLE_TRANSIENT = True

            if ENABLE_TRANSIENT:
                transient_model.train()
                transient_optimizer.zero_grad()

                transient_input = torch.cat((gt_image.unsqueeze(0), image.detach().unsqueeze(0)))
                features = feature_extractor.extract(transient_input).squeeze()

                transient_maps = transient_model(features)

                transient_mask = transient_maps[0].detach()
                if not pipe.disable_dilate:
                    transient_mask = 1-dilate_mask(1-transient_mask, 1)

                mask = LinearSegmentationHead.interpolate(transient_mask, gt_image.shape[1], gt_image.shape[2]).squeeze()
                transient_maps = LinearSegmentationHead.interpolate(transient_maps, gt_image.shape[1], gt_image.shape[2]).squeeze()

                masks_bank[viewpoint_cam.image_name] = mask

                masked_diff = torch.mean(diff.detach() * transient_maps[0])
                mask_reg = torch.abs(1 - transient_maps[0]).mean()
                transient_loss = masked_diff + 0.1 * mask_reg

                if not pipe.disable_consistency:
                    consistency_reg = torch.mean((1 - transient_maps[0]) * (1 - transient_maps[1].detach()))
                    transient_loss += consistency_reg

                transient_loss.backward()
                transient_optimizer.step()
            else:
                mask = masks_bank.get(viewpoint_cam.image_name, torch.ones(gt_image.shape[2:]).to("cuda"))

            Ll1 = torch.mean(diff * mask)
            ssim_value = torch.mean(ssim_value * mask)

        else:
            if viewpoint_cam.mask is not None:
                Ll1 = torch.mean(diff * viewpoint_cam.mask)
                ssim_value = torch.mean(ssim_value * viewpoint_cam.mask)
            else:
                Ll1 = torch.mean(diff)
                ssim_value = torch.mean(ssim_value)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Total variation regularization
        if opt.lambda_tv and iteration > opt.tv_from_iter and iteration < opt.tv_until_iter:
            normalize_depth = lambda x:  (x-x.min())/(x.max()-x.min())
            depth = normalize_depth(render_pkg["depth"])
            tv = anisotropic_total_variation_loss(depth.squeeze())
            loss += opt.lambda_tv * tv

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_tr_loss_for_log = 0.4 * transient_loss + 0.6 * ema_tr_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Transient Loss": f"{ema_tr_loss_for_log:.{7}f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if ENABLE_TRANSIENT:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp, transient_maps, transient_loss, masks_bank)
            else:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp, masks_bank=masks_bank)
            if iteration == saving_iterations[-1]:
                cams = natsorted(scene.getTrainCameras(), key=lambda x: x.image_name)
                rendered_images = []

                for viewpoint_cam in tqdm(cams):
                    rendered_images.append(prep_img(render(viewpoint_cam, gaussians, pipe, bg)["render"]))
                    gt_image = viewpoint_cam.original_image.cuda()

                make_video(rendered_images, scene.model_path, "train", fps=fps)

                if scene.getTestCameras():
                    eval_cams = natsorted(scene.getTestCameras(), key=lambda x: x.image_name)
                    rendered_images = []
                    for cam in tqdm(eval_cams):
                        rendered_images.append(prep_img(render(cam, gaussians, pipe, bg)["render"]))

                    make_video(rendered_images, scene.model_path, "test", fps=fps)
                    evaluate(gaussians, eval_cams, pipe, background, os.path.join(args.model_path, "report.txt"))

                if not pipe.disable_transient:
                    masked_frames = []
                    transient_model.eval()
                    for viewpoint_cam in tqdm(cams):
                        gt_image = viewpoint_cam.original_image.cuda()
                        mask = masks_bank[viewpoint_cam.image_name]
                        masked_img = mask_frame(prep_img(gt_image), mask)
                        masked_frames.append(masked_img)

                    make_video(masked_frames, scene.model_path, "masked", fps=fps)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if not pipe.disable_transient:
                    torch.save(transient_model.state_dict(), scene.model_path + f"/transient_{iteration}.pth")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def evaluate(gaussians, cameras, pipeline, background, path_to_save):
    ssims = []
    psnrs = []

    for idx, view in enumerate(tqdm(cameras)):
        with torch.no_grad():
            rendered_img  = torch.clamp(render(view, gaussians, pipeline, background)["render"], 0.0, 1.0)
            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            ssims.append(ssim(rendered_img, gt_image).mean())
            psnrs.append(psnr(rendered_img, gt_image).mean())
            if (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
    
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    with open(path_to_save, "w") as f:
        f.write("SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        f.write("\n")
        f.write("PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))

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
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, transient_maps=None, transient_loss=None, masks_bank=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        if transient_maps is not None:
            tb_writer.add_scalar('gt_image_dynamic_area', torch.mean(1 - transient_maps[0]).item(), iteration)
            tb_writer.add_scalar('rendered_image_static_area', torch.mean(transient_maps[1]).item(), iteration)

        if transient_loss is not None:
            tb_writer.add_scalar('transient_loss', transient_loss, iteration)


    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        if masks_bank is not None and viewpoint.image_name in masks_bank:
                            mask = masks_bank[viewpoint.image_name]
                            masked_img = mask_frame(prep_img(gt_image), mask)

                            masked_img = masked_img.astype(np.float32) / 255.0
                            masked_img = masked_img.transpose(2, 0, 1)[None]
                            tb_writer.add_images(config['name'] + "_view_{}/mask".format(viewpoint.image_name), masked_img, global_step=iteration)

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

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--fps", type=int, default = 8)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.fps)

    # All done
    print("\nTraining complete.")
