import open3d as o3d
import os

def combine_final_point_cloud_Lidar(source, final, output_path):
    print("Combining filtered point cloud with LiDAR input")

    # Path to the Lidar point cloud
    point_cloud_path1 = os.path.join(source, 'lidar_pointcloud.ply')

    # Read the point clouds
    pcd1 = o3d.io.read_point_cloud(point_cloud_path1)
    pcd2 = o3d.io.read_point_cloud(final)

    # Combine the point clouds
    combined_pcd = pcd1 + pcd2

    # Save the combined point cloud to the specified output path
    o3d.io.write_point_cloud(output_path, combined_pcd)

    return combined_pcd



