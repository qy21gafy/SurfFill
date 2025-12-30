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
import random
import math
import json
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.subdivision import *
from scene.calculate_probablility import *







class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, subdiv, subdivW, subdivH, iteration, subd_params=None, preload = False, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.empty_indices = []
        self.blocks = []

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if (not subdiv) or (subdiv and iteration==-1):
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            else:
                assert False, "Could not recognize scene type!"
        ### For chunk training select appropriate transform json and chunk point cloud depending on iteration
        if (subdiv and iteration>-1):
            if os.path.exists(os.path.join(args.source_path, f"transforms_train{iteration}.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, str(iteration))
            else:
                assert False, "Could not recognize scene type!"

        
        if (not subdiv) or (subdiv and iteration>-1):
            ##### Preload uncertainty images
            self.uncertainmasks = {}
            if preload:
                print("Preloading Uncertainty Masks")
                for cam in scene_info.train_cameras:
                    path = args.source_path
                    uncertainpath = os.path.normpath(os.path.join(path,"uncertainty", os.path.basename(cam.image_path).replace('.jpg', '') + "_pred_alpha.png"))
                    uncertain = cv2.imread(uncertainpath, cv2.IMREAD_GRAYSCALE)
                    normalized_uncertain = uncertain / 255.0
                    maskuncertain = np.ones_like(normalized_uncertain, dtype=np.uint8)
                    uncertainthreshhold = subd_params.uncertainthreshhold 
                    maskuncertain[normalized_uncertain > uncertainthreshhold] = subd_params.uncertainweight 
                    maskuncertain = torch.from_numpy(maskuncertain).cuda()
                    self.uncertainmasks[os.path.basename(cam.image_path).replace('.jpg', '')] = maskuncertain

            ##### Assign camera probability    
            if not load_iteration:
                probabilities = []
                for cam in scene_info.train_cameras:            
                    camera = CameraSub(cam.width, cam.height, cam.T, cam.R, cam.FovX, cam.image_path)    
                    points_tensor = np.stack([np.asarray(scene_info.point_cloud.points)],  axis=0)
                    mask = np.full(len(scene_info.point_cloud.points), False)  
                    maskpost,points = extend_points_in_blocks(mask, points_tensor, camera, subd_params.frustum_length_expand_points_prob, subd_params.random_points_in_frustum)
                    curvs = np.asarray(scene_info.point_cloud.curvatures)[maskpost]
                    probability = calculate_camera_probability(points, curvs, cam.image_path, subd_params.maxprob)
                    probabilities.append(probability)
                total_sum = np.sum(probabilities)

                # Normalize the probabilities
                self.normalized_probabilities = probabilities / total_sum


        ########## Subdiv preprocessing ##############
        if subdiv and iteration==-1:

            if subd_params.recalculate_cuts:
                num_points = np.asarray(scene_info.point_cloud.points).shape[0]
                if num_points>subd_params.desired_points:
                    goal = num_points / subd_params.desired_points
                    dim = int(math.sqrt(goal))
                    subdivW = dim
                    subdivH = dim

            subdivided_blocks = subdivide_box(subd_params.scene_lower_corner[0],subd_params.scene_lower_corner[1],subd_params.scene_lower_corner[2],subd_params.scene_length,subd_params.scene_width,subd_params.scene_height,subdivW,subdivH, subd_params.extend_percentage)
            self.blocks = subdivided_blocks

            points_tensor = np.stack([np.asarray(scene_info.point_cloud.points), np.asarray(scene_info.point_cloud.normals), np.asarray(scene_info.point_cloud.colors)],  axis=0)
            masks,_ = assign_points_to_blocks(points_tensor, subdivided_blocks)

            curv = np.asarray(scene_info.point_cloud.curvatures).copy()
            self.empty_indices = []
            points = {i: [] for i in range(len(subdivided_blocks))}
            i = 0
            for block in subdivided_blocks:
                ### Select cameras belonging to chunk depending on frusta
                camlist = []
                for cam in scene_info.train_cameras:
                    camera = CameraSub(cam.width, cam.height, cam.T, cam.R, cam.FovX, cam.image_path)       
                    if block.contains_points(camera.extrinsic_matrix[:3,3]):
                        camlist.append(camera)
                        masks[i],points[i] = extend_points_in_blocks(masks[i], points_tensor, camera, subd_params.frustum_length_expand_points, subd_params.random_points_in_frustum)
                    else:
                        contain = frustum_block_overlap(camera, block, subd_params.max_cam_distance_to_block, subd_params.random_points_in_frustum, subd_params.frustum_length)
                        if contain:
                            camlist.append(camera)
                data = {
                    "camera_angle_x": camera.fov_radians,
                    "camera_angle_y": camera.fov_radians,
                    "w": camera.w,
                    "h": camera.h,
                    "cx": camera.cx,
                    "cy": camera.cy,
                    "fl_x": camera.fx,
                    "fl_y": camera.fy,
                    "aabb_scale": max(subd_params.scene_length, subd_params.scene_width),
                    "frames": []
                }   
                for image in camlist:
                    frame = {
                        "file_path": image.name,
                        "transform_matrix": image.extrinsic_matrix.tolist()
                    }
                    data["frames"].append(frame)
    
                with open(os.path.join(args.source_path, f"transforms_train{i}.json"), 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                ### Don't train if camlist is empty for a chunk
                if len(camlist)==0:
                    self.empty_indices.append(i)
                i = i  + 1
            empty_indices = save_point_clouds(masks,curv, points, args.source_path, subd_params.drop_point_num) 
            self.empty_indices = self.empty_indices + empty_indices
            return


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, subdiv, block):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        ### Remove Gaussians outside of AABB if subdiv
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), subdiv, block)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]