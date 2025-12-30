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
import time
import torch
import gc
from random import randint
from utils.loss_utils import l1_loss, ssim, VGGPerceptualLoss, vgg_loss
from utils.graphics_utils import patch_offsets, patch_warp
from utils.general_utils import build_scaling_rotation
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image, gradient_map, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, SubdivisionParams, FilteringParams, PreprocessingParams
from scene.combine import *
from preprocess.preprocess_scene import *
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
import torch.nn.functional as F
import yaml
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# For combining the point clouds
root_folders = []

# For measuring time
timepath = ""




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, subdiv, subdivW, subdivH, combine, preprocess, subd_params, combineLidar, filtering_params, preproc_params):
    first_iter = 0
    iterationS = -1
    stop = 1 
    ### Preprocess if wanted to get transforms files and structured reduced point cloud - technically only has to be done once
    if preprocess:
        preprocess_all(args.source_path, preproc_params, dataset.initial_points)
    ### Prepare iteration count depending on training once or iteratively per chunk
    if not subdiv:
        stop = 1
        empty_chunks = []
    else:
        gaussiansPre = GaussianModel(dataset.sh_degree, dataset.max_gaussians)   
        ### Subdivide the initial pointcloud into chunks and select cameras belonging to each
        scenePre = Scene(dataset, gaussiansPre, subdiv, subdivW, subdivH, iterationS, subd_params)
        empty_chunks = scenePre.empty_indices
        blocks = scenePre.blocks
        stop = len(blocks)
        print(empty_chunks)
        print(stop)


    ### Train once or once for every chunk
    while (iterationS < (stop - 1)):

        iterationS += 1
        if iterationS in empty_chunks:
            if combine and iterationS == stop - 1:
                combine_point_clouds(root_folders, opt.iterations, combineLidar, args.source_path)
            continue
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset, iterationS)
        gaussians = GaussianModel(dataset.sh_degree, dataset.max_gaussians)
        scene = Scene(dataset, gaussians, subdiv, subdivW, subdivH, iterationS, subd_params, opt.preload)
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 1, 0]  ### set bg color to green
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        ema_dist_for_log = 0.0
        ema_normal_for_log = 0.0

        ### Add edge, scale logging
        ema_edge_for_log = 0.0
        ema_scale_for_log = 0.0

        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1

        #plt.switch_backend('TkAgg')
        for iteration in range(first_iter, opt.iterations + 1):        

            iter_start.record()

            xyz_lr = gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            

            #### Use the probablilities
            viewpoint_cam = np.random.choice(viewpoint_stack, p=scene.normalized_probabilities)
        
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda()

            ### Add uncertainty mask
            if opt.preload:
                maskuncertain = scene.uncertainmasks.get(viewpoint_cam.image_name, None)
            else:
                uncertainpath = os.path.normpath(os.path.join(os.path.dirname(args.source_path),"uncertainty", viewpoint_cam.image_name + "_pred_alpha.png"))
                uncertain = cv2.imread(uncertainpath, cv2.IMREAD_GRAYSCALE)
                normalized_uncertain = uncertain / 255.0
                maskuncertain = np.ones_like(normalized_uncertain, dtype=np.uint8)
                uncertainthreshhold = subd_params.uncertainthreshhold 
                maskuncertain[normalized_uncertain > uncertainthreshhold] = subd_params.uncertainweight 
                maskuncertain = torch.from_numpy(maskuncertain).cuda()
            gt_image = gt_image * maskuncertain
            image = image * maskuncertain

     

            ### Edge Loss
            edges = gradient_map(image)
            edges_gt = gradient_map(gt_image.to(torch.float32))
            loss_edges = opt.edge_factor *  l1_loss(edges, edges_gt)


            
            # 2DGS image loss
            Ll1 = l1_loss(image, gt_image)              
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

            ###### Scale loss
            if visibility_filter.sum() > 0:
                scale = gaussians.get_scaling[visibility_filter]
                loss_scale = scale.mean() * opt.scale_weight

        
            # regularization
            lambda_normal = opt.lambda_normal if iteration > 3000 else 0.0      
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0        

            ### Add inverse uncertainty mask
            if opt.preload:
                maskinverseuncertain = torch.where(maskuncertain == subd_params.uncertainweight , torch.tensor(0, device=maskuncertain.device), torch.tensor(1, device=maskuncertain.device))
            else:
                maskinverseuncertain = np.zeros_like(normalized_uncertain, dtype=np.uint8)
                maskinverseuncertain[normalized_uncertain < uncertainthreshhold] = 1
                maskinverseuncertain[normalized_uncertain >= uncertainthreshhold] = 0.0
                maskinverseuncertain = torch.from_numpy(maskinverseuncertain).cuda()


            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            ### Only apply normal regularization to flat areas
            rend_normal = rend_normal * maskinverseuncertain
            surf_normal = surf_normal * maskinverseuncertain
            rend_dist = rend_dist * maskinverseuncertain

            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            # Total loss
            total_loss = (1.0/subd_params.uncertainweight) * loss + dist_loss + normal_loss + (1.0/subd_params.uncertainweight) * loss_edges  + loss_scale         
            total_loss.backward(retain_graph=True)

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
                ema_edge_for_log = 0.4 * loss_edges.item() + 0.6 * ema_edge_for_log ### Add edge loss report
                ema_scale_for_log = 0.4 * loss_scale.item() + 0.6 * ema_scale_for_log ### Add scale loss report


                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        "normal": f"{ema_normal_for_log:.{5}f}",
                        "edges": f"{ema_edge_for_log:.{5}f}", ### Add edge loss report
                        "scale": f"{ema_scale_for_log:.{5}f}", ### Add scale loss report
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                if tb_writer is not None:
                    tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    if not subdiv:
                        scene.save(iteration, subdiv, None)
                    else:
                        scene.save(iteration, subdiv, blocks[iterationS])
                    continue

                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, iteration)  
                
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    ### Add noise
                    noise_lr = opt.noiselr               
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)
                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))                
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*noise_lr*xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    

            with torch.no_grad():        
                if network_gui.conn == None:
                    network_gui.try_connect(dataset.render_items)
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                        if custom_cam != None:
                            render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                            net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        metrics_dict = {
                            "#": gaussians.get_opacity.shape[0],
                            "loss": ema_loss_for_log
                            # Add more metrics as needed
                        }
                        # Send the data
                        network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                        if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        # raise e
                        network_gui.conn = None


        ### Combine and filter point cloud high curvature regions before finishing
        if combine and iterationS == stop - 1:
            combine_point_clouds(root_folders, opt, combineLidar, args.source_path, filtering_params)
        ### Clear Caches before moving to next chunk training iteration
        gaussians = None
        scene = None
        torch.cuda.empty_cache()
        gc.collect()

def prepare_output_and_logger(args, iterationS):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    path = args.model_path+str(iterationS)
    global timepath
    timepath = path
    root_folders.append(path)
    args.model_path = args.model_path+str(iterationS)
    print("Output folder: {}".format(path))
    os.makedirs(path, exist_ok = True)
    with open(os.path.join(path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        loss_fn_vgg = VGGPerceptualLoss()
        loss_fn_vgg.to("cuda:0")
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0 ### Add lpips
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += vgg_loss(loss_fn_vgg, image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test))

                ### write results to file
                evalpath = os.path.normpath(os.path.join(timepath, f"evaluationresult{iteration}{config['name']}.txt"))   
                with open(evalpath, "w") as file:
                    file.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test))


                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()



# Update the argument parser's parameters with scene-specific values
def update_params_with_config(args, config, scene):
    scene_params = config.get(scene, {})
    for category, params in scene_params.items():
        for param_name, value in params.items():
            if hasattr(args, param_name):
                setattr(args, param_name, value)

if __name__ == "__main__":
    ### Measure run time
    start_time = time.time()

    ### Add config file
    config_file = "scenes_config.yaml"  
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    subd = SubdivisionParams(parser)
    fi = FilteringParams(parser)
    prep = PreprocessingParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[25_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[25_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    ######### Add subdivision argument
    parser.add_argument('--subdiv', action='store_true', default=False)
    parser.add_argument('--subdivW', type=int, default=2) #vorher 6, 1
    parser.add_argument('--subdivH', type=int, default=1)

    ######## Add combination argument
    parser.add_argument('--combine', action='store_true', default=True)

    ######## Add Preprocess argument
    parser.add_argument('--preprocess', action='store_true', default=True)

    ######## Add final combination argument
    parser.add_argument('--combineLidar', action='store_true', default=True)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.scene_name != "":
        update_params_with_config(args, config, args.scene_name)
        print(vars(args))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.subdiv, args.subdivW, args.subdivH, args.combine, args.preprocess, subd.extract(args), args.combineLidar, fi.extract(args), prep.extract(args))

    end_time = time.time()

    ### Calculate the elapsed time in seconds
    elapsed_time = end_time - start_time

    path = os.path.normpath(os.path.join(timepath, "time", "execution_time.txt"))
    
    os.makedirs(os.path.normpath(os.path.join(timepath, "time")), exist_ok=True)
    with open(path, "w") as file:
        file.write(f"Execution time: {elapsed_time:.6f} seconds\n")
    print(f"Execution time written to {path}")
    print(f"Execution time: {elapsed_time:.6f} seconds\n")
    # All done
    print("\nTraining complete.")