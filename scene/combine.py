import os
import uuid
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scene.filter_points import *
from scene.combineLiDAR import *


def load_point_clouds_from_folders(root_folders, iteration):

    max_sh_degree = 1
    Fxyz = np.empty((0, 3))
    Ffeatures_dc = np.empty((0, 3)) 
    Ffeatures_rest = np.empty((0, 3 * (max_sh_degree + 1) ** 2 - 3)) 
    Fopacity = np.empty((0, 1))
    Fscaling = np.empty((0, 2))
    Frotation = np.empty((0, 4))

    Fcurvature = np.empty((0, 1))
    Fng = np.empty((0, 1))

    for root_folder in root_folders:
        point_cloud_folder = os.path.join(root_folder, "point_cloud")
        
        for folder_name in os.listdir(point_cloud_folder):
            if folder_name.startswith(f"iteration_{iteration}"):
                pcd_path = os.path.join(point_cloud_folder, folder_name, "point_cloud.ply")
                pcd = o3d.io.read_point_cloud(pcd_path)
                plydata = PlyData.read(pcd_path)

                xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                                np.asarray(plydata.elements[0]["y"]),
                                np.asarray(plydata.elements[0]["z"])),  axis=1)
                opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
                curvature = np.asarray(plydata.elements[0]["curvature"])[..., np.newaxis]
                ng = np.asarray(plydata.elements[0]["newgaussian"])[..., np.newaxis]

                features_dc = np.zeros((xyz.shape[0], 3))
                features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
                features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
                features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

                extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
                extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
                assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
                print(len(extra_f_names))
                features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

                scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
                scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
                scales = np.zeros((xyz.shape[0], len(scale_names)))
                for idx, attr_name in enumerate(scale_names):
                    scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

                rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
                rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
                rots = np.zeros((xyz.shape[0], len(rot_names)))
                for idx, attr_name in enumerate(rot_names):
                    rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

                Fxyz = np.concatenate((Fxyz, xyz), axis=0)
                Ffeatures_dc = np.concatenate((Ffeatures_dc, features_dc), axis=0)
                Ffeatures_rest = np.concatenate((Ffeatures_rest, features_extra), axis=0)
                Fopacity = np.concatenate((Fopacity, opacities), axis=0)
                Fcurvature = np.concatenate((Fcurvature, curvature), axis=0)
                Fng = np.concatenate((Fng, ng), axis=0)
                Fscaling = np.concatenate((Fscaling, scales), axis=0)
                Frotation = np.concatenate((Frotation, rots), axis=0)


    return Fxyz, Ffeatures_dc, Ffeatures_rest, Fopacity, Fscaling, Frotation, Fcurvature, Fng

def construct_list_of_attributes(Ffeatures_dc, Ffeatures_rest, Fscaling, Frotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(Ffeatures_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(Ffeatures_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    l.append('curvature')
    l.append('newgaussian')
    for i in range(Fscaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(Frotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(path, Fxyz, Ffeatures_dc, Ffeatures_rest, Fopacity, Fscaling, Frotation, Fcurvature, Fng):

    xyz = Fxyz
    normals = np.zeros_like(xyz)
    f_dc = Ffeatures_dc
    f_rest = Ffeatures_rest
    opacities = Fopacity
    scale = Fscaling
    rotation = Frotation

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(Ffeatures_dc, Ffeatures_rest, Fscaling, Frotation)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, Fcurvature, Fng, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def combine_point_clouds(root_folders, opt, combineLidar, source_path, filtering_params):
    unique_str = str(uuid.uuid4())
    output_folder = os.path.join("./output/", unique_str[0:10])
    print("Combining Point Clouds...")
    output_path = os.path.join(output_folder, "combined_point_cloud.ply")
    os.makedirs(output_folder, exist_ok = True)
        # Load all point clouds
    Fxyz, Ffeatures_dc, Ffeatures_rest, Fopacity, Fscaling, Frotation, Fcurvature, Fng = load_point_clouds_from_folders(root_folders, opt.iterations)
    save_ply(output_path, Fxyz, Ffeatures_dc, Ffeatures_rest, Fopacity, Fscaling, Frotation, Fcurvature, Fng)
    print(f"Combined point cloud saved to {output_path}")
    print("Filtering Point Cloud...")
    thin_output_file_path = filterThin(output_path,  os.path.join(output_folder, "filtered_point_cloud.ply"), source_path, filtering_params)

    if combineLidar:
        combine_final_point_cloud_Lidar(source_path, thin_output_file_path, os.path.join(output_folder, "updatedLidar.ply"))



