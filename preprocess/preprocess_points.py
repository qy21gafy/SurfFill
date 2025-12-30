import os
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import torch
from simple_knn._C import distCUDA2

def load_point_cloud(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                    np.asarray(plydata.elements[0]["ny"]),
                    np.asarray(plydata.elements[0]["nz"])),  axis=1)
    colors = np.stack((np.asarray(plydata.elements[0]["red"]),
                    np.asarray(plydata.elements[0]["green"]),
                    np.asarray(plydata.elements[0]["blue"])),  axis=1)
    return xyz, normals, colors


def save_ply(path, xyz, normals, colors, curvatures):

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('curvature', 'f4')]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, colors, curvatures), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def separate_structures(curvatures, curvature_threshold):
    # Separate points based on the curvature threshold
    thin_structure_indices = np.where(((curvatures > curvature_threshold) & (curvatures > 0)) | ((curvatures < -curvature_threshold) & (curvatures < 0)))[0]
    flat_area_indices = np.where(((curvatures <= curvature_threshold) & (curvatures > 0)) | ((curvatures >= -curvature_threshold) & (curvatures < 0)))[0]
    
    return thin_structure_indices, flat_area_indices

def reduceStructured(input_file_path, output_file_path, num_random_points, curvature_threshold, testscene=False): 
    # Load the point cloud
    xyz, normals, colors = load_point_cloud(input_file_path)   


    if testscene:
        curvatures = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
        curvatures = np.clip(2*curvature_threshold - ((curvatures.cpu().numpy()[..., np.newaxis])*6000), a_min=0.0000001, a_max=None)
        # Separate structures
        thin_structure_indices, flat_area_indices  = separate_structures(curvatures, curvature_threshold)
        flat_area_indices = torch.tensor(flat_area_indices)

        if len(thin_structure_indices)>num_random_points:
            thin_structure_indices = torch.tensor(thin_structure_indices)
            factor = 0.5
            num_random_pointsFlat = int(factor * num_random_points)

            # Randomly select indices without replacement
            selected_indices_flat = torch.randperm(len(flat_area_indices))[:num_random_pointsFlat]
            selected_indices_thin = torch.randperm(len(thin_structure_indices))[:(num_random_points-len(selected_indices_flat))]

            # Get the subset of indices
            flat_area_indices_red = flat_area_indices[selected_indices_flat]
            thin_structure_indices_red = thin_structure_indices[selected_indices_thin]
        else:
            # Randomly select indices without replacement
            selected_indices = torch.randperm(len(flat_area_indices))[:(num_random_points-len(thin_structure_indices))]

            # Get the subset of flat_area_indices
            flat_area_indices_red = flat_area_indices[selected_indices]
            thin_structure_indices_red = torch.tensor(thin_structure_indices)
    
        # Combine the point clouds
        combined_indices = torch.cat((thin_structure_indices_red, flat_area_indices_red))
    else:
        curvatures = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
        curvatures = curvatures.cpu().numpy()[..., np.newaxis] * 1000 #5700 for scannet 1ae, 5200 for 45, 3500 for f
        # Separate structures
        thin_structure_indices, flat_area_indices = separate_structures(curvatures, curvature_threshold)
        flat_area_indices = torch.tensor(flat_area_indices)

        # Randomly select indices without replacement
        selected_indices = torch.randperm(len(flat_area_indices))[:(num_random_points-len(thin_structure_indices))]

        # Get the subset of flat_area_indices
        flat_area_indices_red = selected_indices
    
        # Combine the point clouds
        combined_indices = torch.cat((torch.tensor(thin_structure_indices), flat_area_indices_red))

    xyz_selected = xyz[combined_indices]
    normals_selected = normals[combined_indices]
    colors_selected = colors[combined_indices]
    curvatures_selected = curvatures[combined_indices]

    # Save the combined point cloud
    output = os.path.join(output_file_path,"points3d.ply")
    save_ply(output,xyz_selected, normals_selected, colors_selected, curvatures_selected)
     

    print(f"Combined point clouds saved to {output}")   


