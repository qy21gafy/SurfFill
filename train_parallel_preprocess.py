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


import uuid
from scene import Scene, GaussianModel
from preprocess.preprocess_scene import *
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def pre_training(dataset, subd_params, source_path, subdiv, subdivW, subdivH, preprocess, preproc_params):

    ### Preprocess if wanted to get transforms files, uncertainty masks and structured reduced point cloud - technically only has to be done once
    if preprocess:
        print("Preprocessing")
        preprocess_all(source_path, preproc_params, dataset.initial_points)
    ### Prepare iteration count depending on training once or iteratively per chunk
    gaussiansPre = GaussianModel(dataset.sh_degree, dataset.max_gaussians)   
    ### Subdivide the initial pointcloud into chunks and select cameras belonging to each
    scenePre = Scene(dataset, gaussiansPre, subdiv, subdivW, subdivH, -1, subd_params)
    empty_chunks = scenePre.empty_indices
    blocks = scenePre.blocks
    stop = len(blocks)
    print(empty_chunks)

    unique_str = str(uuid.uuid4())

    return unique_str, stop, empty_chunks, blocks

