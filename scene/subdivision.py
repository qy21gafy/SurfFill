import os
from plyfile import PlyData, PlyElement
import numpy as np

##################  Subdivided Processing ########################
class Box:
    def __init__(self, x, y, z, length, width, height, extend_percentage):
        self.x = x  
        self.y = y 
        self.z = z  
        self.length = length
        self.width = width
        self.height = height
        self.extend_percentage = extend_percentage

    def __repr__(self):
        return f"Box(x={self.x}, y={self.y}, z={self.z}, length={self.length}, width={self.width}, height={self.height})"

    def contains(self, points):
        within_x = (self.x -  self.extend_percentage * self.length <= points[0,:, 0]  ) & (points[0,:, 0] < self.x + self.length + self.extend_percentage * self.length)           ####### Slightly increase box size for better edge regions
        within_y = (self.y - self.extend_percentage * self.width <= points[0,:, 1] ) & (points[0,:, 1] < self.y + self.width + self.extend_percentage * self.width)
        within_z = (self.z - self.extend_percentage * self.height <= points[0,:, 2] ) & (points[0,:, 2] < self.z + self.height + self.extend_percentage * self.height)
        return within_x & within_y & within_z

    # No extension
    def contains_points(self, points):
        within_x = (self.x <= points[0]) & (points[0] < self.x + self.length)
        within_y = (self.y <= points[1]) & (points[1] < self.y + self.width)
        within_z = (self.z <= points[2]) & (points[2] < self.z + self.height)

        return within_x & within_y & within_z


class CameraSub:
    def __init__(self, image_width, image_height, T, R, fov, name):
        self.name = name
        self.T = T
        self.R = R
        self.fov = fov
        R_transpose = np.transpose(R)  # Transpose R if it's stored transposed
        c2w = np.eye(4)
        c2w[:3, :3] = R_transpose
        c2w[:3, 3] = T

        # Get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        w2c[:3, 1:3] *= -1

        self.extrinsic_matrix = w2c
        self.cx = image_width/2
        self.cy = image_height/2
        self.w = image_width
        self.h = image_height
        self.aspect_ratio = self.w / self.h        

        self.fov_radians = fov

        # Compute focal lengths
        self.fx = image_width / (2 * np.tan(self.fov_radians / 2))
        self.fy = image_height / (2 * np.tan(self.fov_radians / 2))
        
        
        self.intrinsic_matrix = np.array([
        [self.fx, 0, self.cx],
        [0, self.fy, self.cy],
        [0, 0, 1]
        ])




    def compute_frustum_corners(self, random_points_in_frustum, frustum_length):

        fov_rad = self.fov_radians
        near = 0.1
        far = frustum_length
        aspect_ratio = self.aspect_ratio
        # Calculate the height and width of the near and far planes
        h_near = 2 * np.tan(fov_rad / 2) * near
        w_near = h_near * aspect_ratio
        h_far = 2 * np.tan(fov_rad / 2) * far
        w_far = h_far * aspect_ratio

        # Define the corners of the near and far planes in the camera's local space
        near_plane_corners = [
            np.array([-w_near / 2, -h_near / 2, near]),
            np.array([ w_near / 2, -h_near / 2, near]),
            np.array([ w_near / 2,  h_near / 2, near]),
            np.array([-w_near / 2,  h_near / 2, near])
        ]

        far_plane_corners = [
            np.array([-w_far / 2, -h_far / 2, far]),
            np.array([ w_far / 2, -h_far / 2, far]),
            np.array([ w_far / 2,  h_far / 2, far]),
            np.array([-w_far / 2,  h_far / 2, far])
        ]
    
        # Combine near and far plane corners
        all_corners = near_plane_corners + far_plane_corners

        # Transform each corner separately
        frustum_corners_world = [calculate_frustum_corner(self.extrinsic_matrix[:3,3], self.extrinsic_matrix[:3,:3], corner) for corner in all_corners]      

        random_points = generate_random_points_in_frustum(random_points_in_frustum, self.extrinsic_matrix[:3,3], self.extrinsic_matrix[:3,:3], self.fov, aspect_ratio, near, far)
        
        return frustum_corners_world, random_points

def generate_random_points_in_frustum(num_points, camera_position, camera_rotation_matrix, fov, aspect_ratio, near, far):
    # Convert FOV to radians
    fov_rad = np.radians(fov)
    
    # Generate random points in normalized frustum space
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    z = np.random.uniform(near, far, num_points)
    
    # Calculate the scale factors for x and y 
    scale_x = z * np.tan(fov_rad / 2) * aspect_ratio
    scale_y = z * np.tan(fov_rad / 2)
    
    # Scale x and y by the calculated factors
    x = x * scale_x
    y = y * scale_y
    
    # Create the points in view space
    points_view = np.vstack((x, y, z)).T
    
    tmp = camera_rotation_matrix.copy()
    tmp[:3, 1:3] *= -1

    # Transform points from view space to world space
    points_world = np.dot(points_view, tmp.T) + camera_position
    
    return points_world

def calculate_frustum_corner(position, rotation_matrix, corner):
    # Transform the corner by the rotation matrix and translate by the camera position
    tmp = rotation_matrix.copy()
    tmp[:3, 1:3] *= -1
    transformed_corner = np.dot(tmp, corner) + position
    return transformed_corner


def frustum_block_overlap(camera, block, max_cam_distance_to_block, random_points_in_frustum, frustum_length):
    # Compute camera frustum corners in world coordinates
    frustum_corners, points = camera.compute_frustum_corners(random_points_in_frustum, frustum_length)

    # Check if frustum overlaps with block and if camera is close enough
    for point in points:
        if block.contains_points(point) and np.linalg.norm(point - camera.extrinsic_matrix[:3,3]) < max_cam_distance_to_block:
            return True


def subdivide_box(x, y, z, length, width, height, subdivisions_length, subdivisions_width, extend_percentage):
    sub_length = length / subdivisions_length
    sub_width = width / subdivisions_width
    
    # Create the subdivided blocks
    blocks = []
    for i in range(subdivisions_length):
        for j in range(subdivisions_width):
            new_x = x + i * sub_length
            new_y = y + j * sub_width
            new_z = z  # no subdivision along height
            blocks.append(Box(new_x, new_y, new_z, sub_length, sub_width, height, extend_percentage))
    return blocks

def assign_points_to_blocks(points, blocks):
    masks = {i: [] for i in range(len(blocks))}
    points_in_blocks = {i: [] for i in range(len(blocks))}
    for i, block in enumerate(blocks):
        masks[i] = block.contains(points)
        points_in_blocks[i] = points[:,masks[i],:]
    return masks, points_in_blocks

def extend_points_in_blocks(mask, points, camera, frustum_length, random_points_frustum):
    frustum_corners, _ = camera.compute_frustum_corners(random_points_frustum, frustum_length)

    near_plane_corners = frustum_corners[:4]
    far_plane_corners = frustum_corners[4:]

    def compute_plane(p1, p2, p3, reverse_normal=False):
        v1 = p1 - p2
        v2 = p3 - p2
        normal = np.cross(v2, v1)
        normal = normal / np.linalg.norm(normal)
        
        if reverse_normal:
            normal = -normal
        d = -np.dot(normal, p1)
        return normal, d
    
    near_plane = compute_plane(near_plane_corners[0], near_plane_corners[1], near_plane_corners[2])
    far_plane = compute_plane(far_plane_corners[0], far_plane_corners[1], far_plane_corners[2], True)
    left_plane = compute_plane(near_plane_corners[0], far_plane_corners[0], far_plane_corners[3], True)
    right_plane = compute_plane(near_plane_corners[1], far_plane_corners[1], far_plane_corners[2])
    top_plane = compute_plane(near_plane_corners[3], far_plane_corners[3], far_plane_corners[2], True)
    bottom_plane = compute_plane(near_plane_corners[0], far_plane_corners[0], far_plane_corners[1])
    

    normals = np.array([near_plane[0], far_plane[0], left_plane[0], right_plane[0], top_plane[0], bottom_plane[0]])  
    d_values = np.array([near_plane[1], far_plane[1], left_plane[1], right_plane[1], top_plane[1], bottom_plane[1]]) 

    dot_products = np.dot(points[0], normals.T)  
    distances = dot_products + d_values  

    inside_frustum = np.all(distances >= 0, axis=1)

    mask[inside_frustum] = True

    points_in_block = points[:,mask,:]
    return mask, points_in_block
    


def save_point_clouds(masks, curvs, points_ready, output_dir, drop_point_num):
    empty_indices = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for block_index,points in points_ready.items():
        xyz_selected = points[0]
        normals_selected = points[1]
        colors_selected = points[2]
        curvatures_selected = np.asarray(curvs[masks[block_index]])[..., np.newaxis]
        ### Only train chunks with a certain amount of points inside
        if xyz_selected.shape[0] > drop_point_num:            
            num_points = xyz_selected.shape[0]  
            print(num_points)           
            # Save PointCloud to file
            filename = os.path.join(output_dir, f"points3d{block_index}.ply")
            save_ply(filename, xyz_selected, normals_selected, colors_selected, curvatures_selected)
        else:
            empty_indices.append(block_index)
    return empty_indices

def save_ply(path, xyz, normals, colors, curvatures):

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('curvature', 'f4')]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, colors*255, curvatures), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)



