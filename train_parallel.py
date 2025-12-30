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
from train_single import *
from train_parallel_preprocess import *
import sys
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, SubdivisionParams, FilteringParams, PreprocessingParams
from train import update_params_with_config
from scene.combine import *
from preprocess.preprocess_scene import *
import subprocess
import time
import os
import yaml
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def submit_job(script, params):
    """Submit a job using sbatch and return the job ID."""    
    slurm_args = ["sbatch", script] + params
    try:
        result = subprocess.run(slurm_args, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error when submitting a job: {e}")
        sys.exit(1)
    # Extract job ID from sbatch output
    job_id = result.stdout.strip().split()[-1]
    print(f"submitted job {job_id}")

    return job_id

def is_job_finished(job_id):
    """Check if the job has finished using sacct."""
    result = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader', '--parsable2'], capture_output=True, text=True)
    # Get job state
    job_state = result.stdout.split('\n')[0]
    return job_state if job_state in {'COMPLETED', 'FAILED', 'CANCELLED'} else ""



def training_parallel(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, subdiv, subdivW, subdivH, combine, preprocess, subd_params, combineLidar, filtering_params, preproc_params):
    root_folders = []

    unique_str, n_chunks, empty_chunks, blocks = pre_training(dataset, subd_params, args.source_path, subdiv, subdivW, subdivH, preprocess, preproc_params)

    for i in range(n_chunks):
        if i in empty_chunks:
            i+=1
            continue
        path = os.path.join("./output/", unique_str[0:10], str(i))
        root_folders.append(path)

    print(root_folders)
    
    slurm_script = "train_chunk.slurm"    
    jobs = []

    i = 0
    for block in blocks:
        if i in empty_chunks:
            i+=1
            continue
        params = [args.source_path, unique_str, str(i), str(blocks[i].x), str(blocks[i].y), str(blocks[i].z), str(blocks[i].length), str(blocks[i].width), str(blocks[i].height) ]
        print(f"Starting job for chunk {i} with params: {params}")
        job = submit_job(slurm_script, params)
        jobs.append(job)
        i+=1


    all_finished = False
    all_status = []
    last_count = 0
    print(f"Waiting for chunks to be trained in parallel ...")

    while not all_finished:

        all_status = [is_job_finished(id) for id in jobs if is_job_finished(id) != ""]
        if last_count != all_status.count("COMPLETED"):
            last_count = all_status.count("COMPLETED")
            print(f"processed [{last_count} / {n_chunks} chunks].")

        all_finished = len(all_status) == len(jobs)
    
        if not all_finished:
            time.sleep(10) 
        
    if not all(status == "COMPLETED" for status in all_status):
        print("At least one job failed or was cancelled, check at error logs.")


    if combine:
        combine_point_clouds(root_folders, opt, combineLidar, args.source_path, filtering_params)
    return unique_str


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 25_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 25_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    ######### Add subdivision argument
    parser.add_argument('--subdiv', action='store_true', default=True)
    parser.add_argument('--subdivW', type=int, default=2) 
    parser.add_argument('--subdivH', type=int, default=2)

    ######## Add combination argument
    parser.add_argument('--combine', action='store_true', default=True)

    ######## Add Preprocess argument
    parser.add_argument('--preprocess', action='store_true', default=False)

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
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    unique_str = training_parallel(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.subdiv, args.subdivW, args.subdivH, args.combine, args.preprocess, subd.extract(args), args.combineLidar, fi.extract(args), prep.extract(args))

    end_time = time.time()

    ### Calculate the elapsed time in seconds
    elapsed_time = end_time - start_time

    path = os.path.normpath(os.path.join("./output/", unique_str[0:10], "time", "execution_time.txt"))
    os.makedirs(os.path.normpath(os.path.join("./output/", unique_str[0:10], "time")), exist_ok=True)

    with open(path, "w") as file:
        file.write(f"Execution time: {elapsed_time:.6f} seconds\n")
    print(f"Execution time written to {path}")
    # All done
    print("\nTraining complete.")