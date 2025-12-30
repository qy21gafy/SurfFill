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

from argparse import ArgumentParser, Namespace
import sys
import os



class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Scene name to load configuration for, one of: attic, kitchen, museum, caterpillar, meetingroom, truck, courthouse, navvisattic
        self.scene_name = ""
        # Number of filtered points after preprocessing
        self.initial_points = 500000
        # Max Gaussians
        self.max_gaussians = 2800000
        self.sh_degree = 1
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1 
        self._white_background = False # Use green background
        self.data_device = "cuda"
        self.eval = True
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']  
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

### Added for subdivision
class SubdivisionParams(ParamGroup):
    def __init__(self, parser):
        self.recalculate_cuts = False
        self.extend_percentage = 0.9
        self.desired_points = 2000000
        self.drop_point_num = 50000
        self.scene_length = 12.9
        self.scene_width = 12.6
        self.scene_height = 3.8
        self.max_cam_distance_to_block = 2 
        self.random_points_in_frustum = 80
        self.frustum_length = 10
        self.frustum_length_expand_points = 3
        self.frustum_length_expand_points_prob = 7
        self.scene_lower_corner = [-4.8, -6.2, -1.1]
        self.maxprob = 0.1
        self.uncertainthreshhold = 0.2
        self.uncertainweight = 2.0 
        super().__init__(parser, "Subdivision Parameters")

class FilteringParams(ParamGroup):
     def __init__(self, parser):
        self.scalethreshold = 3.0
        self.curvature_threshold = 0.04
        self.opacitythreshold = 0.007
        self.distancerounds = 4
        self.nn_points_close = 1
        self.nn_radius_close = 0.05
        self.nn_points_far = 30
        self.nn_radius_far = 0.5
        self.distance_lidar_min = 0.1 
        self.distace_lidar_max = 3
        self.distance_lidar_neighbors = 5
        self.points_per_gaussian = 0
        self.upsample_interpolated = False
        super().__init__(parser, "Filtering Parameters")


class PreprocessingParams(ParamGroup):
     def __init__(self, parser):
        self.testscene = False
        self.testsceneDach = False
        self.tnt = False
        self.generate_maps = True
        self.w = 500
        self.h = 500
        self.curvature_threshold_preprocessing = 0.04
        self.fov = 90
        super().__init__(parser, "Preprocessing Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 25_000
        self.position_lr_init = 0.00004
        self.position_lr_final = 0.000001
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05        
        self.scaling_lr = 0.002
        self.rotation_lr = 0.001

        self.percent_dense = 0.01
        self.lambda_dssim = 0.4 # Use higher dssim percentage for more fine details and edges + more gaussians
        self.lambda_dist = 0.01
        self.lambda_normal = 0.02
        self.opacity_cull = 0.01

        self.densification_interval = 100
        self.opacity_reset_interval = 7550
        self.densify_from_iter = 100
        self.densify_until_iter = 30_000
        self.densify_grad_threshold = 0.00013


        ### Added parameters
        ### Curvature
        self.curvature_lr = 0.05
        ### Start Gaussian
        self.newgaussian_lr = 0
        ### Weight of Edge loss
        self.edge_factor = 0.4
        ### Scale loss weight
        self.scale_weight = 1.0
        
        self.noiselr = 5e5
        self.preload = True

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
