from urllib.request import ProxyBasicAuthHandler
import open3d as o3d
import os
import numpy as np
import re

def calculate_camera_probability(points, curvatures, path, maxprob, curvature_threshhold=0.035):
    curvature_values = curvatures

    # Count the number of points above the curvature threshold
    points_above_threshold = np.sum(curvature_values > curvature_threshhold)

    # Total number of points (N)
    total_points = curvature_values.size
    # Calculate the probability as the ratio of points above the threshold
    if total_points == 0:
        probability = maxprob
    else:
        probability = points_above_threshold / total_points
        probability = max(probability, maxprob) 
    if "+0" in path:
        probability = probability / 3

    return probability
    



