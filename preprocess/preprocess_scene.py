import json
import os
import numpy as np
import time
from PIL import Image
from preprocess.preprocess_points import *
from preprocess.uncertainty.test import *
from scipy.spatial.transform import Rotation as R

def preprocess_all(input_file_path, preproc_params, num_random_points=2000000): 

    print("Reducing Point Cloud....")
    start_time = time.time()
    reduceStructured(os.path.join(input_file_path,"lidar_pointcloud.ply"), input_file_path, num_random_points, preproc_params.curvature_threshold_preprocessing, preproc_params.testscene or preproc_params.testsceneDach)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time Point Cloud Reduction: {elapsed_time:.6f} seconds\n")

    if preproc_params.generate_maps:
        dim = 1024
        if preproc_params.tnt:
            dim = 1080
            generate_uncertainty_maps(os.path.join(os.path.dirname(input_file_path),"images"), dim, 1920)           
        else:
            generate_uncertainty_maps(os.path.join(os.path.dirname(input_file_path),"images"), dim, dim)

        if dim != preproc_params.w:
            uncertainty_dir = os.path.join(os.path.dirname(input_file_path), "uncertainty")
            imagesmultiple_dir = os.path.join(os.path.dirname(input_file_path), "images")

            # Get the target dimensions from the first image in imagesmultiple directory
            target_width, target_height = None, None
            for img_file in os.listdir(imagesmultiple_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):  
                    with Image.open(os.path.join(imagesmultiple_dir, img_file)) as img:
                        target_width, target_height = img.size
                        break 

            if target_width and target_height:
                # Iterate over all images in the uncertainty directory
                for img_file in os.listdir(uncertainty_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(uncertainty_dir, img_file)
            
                        with Image.open(img_path) as img:
                            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                            resized_img.save(img_path)  
