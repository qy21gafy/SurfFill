import os
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import torch
from simple_knn._C import distCUDA2
from scipy.spatial import cKDTree
import math

def load_point_cloud(path):

    max_sh_degree = 1

    pcd_path = path
    pcd = o3d.io.read_point_cloud(pcd_path)
    plydata = PlyData.read(pcd_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])

    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    quats = np.stack((np.asarray(plydata.elements[0]["rot_1"]),
                          np.asarray(plydata.elements[0]["rot_2"]),
                          np.asarray(plydata.elements[0]["rot_3"]),
                          np.asarray(plydata.elements[0]["rot_0"])), axis=1)

    curvs = np.asarray(plydata.elements[0]["curvature"])

    ng = np.asarray(plydata.elements[0]["newgaussian"])

    return pcd, xyz, scales, curvs, quats, ng, opacities



def filter_curvature(curvatures, curvature_threshold): 
    # Separate points based on the curvature threshold
    thin_structure_indices = ((curvatures > curvature_threshold) & (curvatures > 0)) | ((curvatures < -curvature_threshold) & (curvatures < 0))
    
    return thin_structure_indices

def filter_opacities(opacities, opacities_threshold): 
    # Separate points based on the curvature threshold
    thin_structure_indices = opacities > opacities_threshold
    
    return thin_structure_indices


def filter_scale(scales, scalethreshold):

    within_x = (-scalethreshold > scales[:, 0]) 
    within_y = (-scalethreshold > scales[:, 1]) 

    return within_x & within_y 

def generate_interpolated_points(points, num_points, num_neighbors):
    tree = cKDTree(points)
    _, indices = tree.query(points, k=num_neighbors)  

    random_points = [] 

    for i, point in enumerate(points):
        neighbor_indices = indices[i, 1:num_neighbors]  
        selected_neighbor_index = np.random.choice(neighbor_indices) 
        neighbor_point = points[selected_neighbor_index]

        for _ in range(num_points):
            alpha = np.random.rand()

            random_point = (1 - alpha) * point + alpha * neighbor_point
            random_points.append(random_point)


    random_points = np.array(random_points)
    return random_points



def generate_points(xyz, scales, rots, num_points):
    
    scales = np.exp(scales)
    num_ellipsoids = xyz.shape[0]

    mean = 0
    std_dev = 1
    theta = np.random.normal(loc=mean, scale=std_dev, size=(num_ellipsoids, num_points))
    phi = np.random.normal(loc=mean, scale=std_dev, size=(num_ellipsoids, num_points))

    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)


    x_ellipsoids = scales[:, 0][:, np.newaxis] * x_sphere
    y_ellipsoids = scales[:, 1][:, np.newaxis] * y_sphere
    z_ellipsoids = np.zeros(y_ellipsoids.shape)

    points = np.stack((x_ellipsoids, y_ellipsoids, z_ellipsoids), axis=-1)  

    if num_points > 0:
        norm = np.linalg.norm(rots, axis=-1, keepdims=True)
        normalized_q = rots / norm
        rotations = R.from_quat(normalized_q)
        rotation_matrices = rotations.as_matrix()  
    
        rotated_points = np.einsum('ijk,ilk->ijl', rotation_matrices, points) 
        rotated_points = rotated_points.transpose(0, 2, 1)

        final_points = rotated_points + xyz[:, np.newaxis, :]  
    
        return final_points
    else:
        return points


def remove_radius_outlier(points, nb_points, radius):

    tree = cKDTree(points)

    distances, _ = tree.query(points, k=nb_points+1)  

    max_distances = np.max(distances[:, 1:], axis=1)

    inlier_mask = max_distances < radius

    filtered_points = points[inlier_mask]
    inlier_indices = np.where(inlier_mask)[0]

    return filtered_points, inlier_indices


def compute_distances(source_pc, target_pc):
    dists = target_pc.compute_point_cloud_distance(source_pc)
    return np.asarray(dists)

def filter_points_based_on_threshold(source_pc, distances, threshold, num_neighbors, threshold_max):

    num_points = len(source_pc.points)
    mask = np.zeros(num_points, dtype=bool)
    kd_tree = o3d.geometry.KDTreeFlann(source_pc)
    
    for i, dist in enumerate(distances):
        # Find the n nearest neighbors of the point
        [_, idx, _] = kd_tree.search_knn_vector_3d(source_pc.points[i], num_neighbors)
        
        # Sum the distances of the point and its n nearest neighbors
        sum_distances = dist + np.sum(distances[idx])
        
        # Set the mask to True if the sum of distances is greater than the threshold
        if sum_distances > threshold:
            mask[i] = True
            if sum_distances > threshold_max:
                mask[i] = False

    
    return mask


    

def filterThin(input_file_path, thin_output_file_path,  source_path, filtering_params): 
    # Load the point cloud
    pcd, xyz, scales, curvs,  rots, ng, opacities = load_point_cloud(input_file_path)
    print(pcd)
    curvatures = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
    curvatures = curvatures.cpu().numpy() * 100
    maskscale = filter_scale(scales, filtering_params.scalethreshold)
    maskcurv = filter_curvature(curvatures, filtering_params.curvature_threshold)
    imaskcurv = [not val for val in maskcurv.tolist()]
    maskf = np.logical_and(maskscale, imaskcurv)

    # filter opacities
    masko = filter_opacities(np.exp(opacities), filtering_params.opacitythreshold) 
    maskf = np.logical_and(maskf, masko)
    
    pcdFiltered = np.asarray(pcd.points)[maskf]
    scales = scales[maskf]
    rots = rots[maskf]
    ng = ng[maskf]
    print(scales.shape)
    
    print("Refining point cloud...")

    ######### Perform radius outlier removal 
    
    for i in range(1,filtering_params.distancerounds):
        dist = torch.clamp_min(distCUDA2(torch.from_numpy(pcdFiltered).float().cuda()), 0.00000001)
        maskcurv2 = filter_curvature(dist, 0.005/(i*2)) 
        mask2 = [not val for val in maskcurv2.tolist()]
        pcdFiltered = np.asarray(pcdFiltered)[mask2]
        scales = scales[mask2]
        rots = rots[mask2]
        ng = ng[mask2]
        print(scales.shape)

    print("Removing outliers...")


    filtered_points1, ind1 = remove_radius_outlier(pcdFiltered, filtering_params.nn_points_close, filtering_params.nn_radius_close) 
    scales = scales[ind1]
    rots = rots[ind1]
    ng = ng[ind1]
    print(np.asarray(filtered_points1).shape)

    filtered_points2, ind2 = remove_radius_outlier(filtered_points1, filtering_params.nn_points_far, filtering_params.nn_radius_far) 
    scales = scales[ind2]
    rots = rots[ind2]
    ng = ng[ind2]
    print(np.asarray(filtered_points2).shape)


    maskng = ng == 1
    scales = scales[maskng]
    rots = rots[maskng]
    filtered_points2 = filtered_points2[maskng]

    # Filter based on distance
    # Determine the parent folder of the given folder
    parent_folder_path = source_path
    lidar_path = os.path.join(parent_folder_path, 'lidar_pointcloud.ply')
    source_pc = o3d.io.read_point_cloud(lidar_path)
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(filtered_points2)
    distances = compute_distances(source_pc, target_pc)
    mask_far = filter_points_based_on_threshold(target_pc, distances, filtering_params.distance_lidar_min, filtering_params.distance_lidar_neighbors, filtering_params.distace_lidar_max) # (target_pc, distances, 0.1, 5, 3) rest, (target_pc, distances, 0.5, 5, 70) Courthouse, (target_pc, distances, 0.02, 5, 1) truck
    far_points = target_pc.select_by_index(np.where(mask_far)[0])
    scales = scales[mask_far]
    rots = rots[mask_far]
    

    # Adding more points then just the center gaussian
    more_points = generate_points(np.asarray(far_points.points), scales, rots, filtering_params.points_per_gaussian)
    if filtering_params.upsample_interpolated:
        more_points = generate_interpolated_points(far_points.points, filtering_params.points_per_gaussian, 5)
    more_points = more_points.reshape(-1, 3)
    more_points = np.concatenate((more_points, np.asarray(far_points.points)), axis=0)
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(more_points)
    print(np.asarray(final_pcd.points).shape)

    # Save the separated point clouds
    o3d.io.write_point_cloud(thin_output_file_path,  final_pcd)
  
    print(f"Thin structures saved to {thin_output_file_path}")

    return thin_output_file_path

