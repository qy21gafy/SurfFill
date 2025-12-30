# SurfFill: Completion of LiDAR Point Clouds via Gaussian Surfel Splatting

[Project page](https://lfranke.github.io/surffill/) | [Paper](https://arxiv.org/abs/2512.03010) | [Video](https://www.youtube.com/watch?v=OS3q5OWT-sg) | [Surfel Rasterizer (CUDA)](https://github.com/hbb1/diff-surfel-rasterization) | [Surfel Rasterizer (Python)](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing) | [SIBR Viewer Pre-built for Windows](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing)<br>

![Teaser image](assets/header.pdf)

This repo contains the official implementation for the paper "SurfFill: Completion of LiDAR Point Clouds via Gaussian Surfel Splatting". Our work improves the completeness of LiDAR point clouds by combining the initial set of LiDAR points with selected points reconstructed using 2D Gaussian Splatting.

## Installation
Tested with CUDA 12.8 and Torch 2.9.1+cu128
```bash
# download
git clone https://github.com/hbb1/surffill.git --recursive

enter that folder
pipenv --python 3.10
pipenv shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ffmpeg pillow open3d mediapy lpips scikit-image tqdm trimesh plyfile opencv-python
enter subfolders
submodules/diff-surfel-rasterization
submodules/simple-knn
in each run 
pipenv run python setup.py install
```
Download scannet.pt from https://huggingface.co/ckpt/ControlNet-v1-1/blob/2e01957f9da7799e4c669811c08d1617bdae51bd/scannet.pt and place it into surffill/checkpoints/scannet.pt

## Training
To train a scene, simply use
```bash
PYTHONPATH=. python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

### Quick Examples
Assuming you have downloaded [Caterpillar](https://jonbarron.info/mipnerf360/), simply use
```bash
PYTHONPATH=. python train.py -s <path to caterpillar dataset>/<demo> --scene_name caterpillar

**Custom Dataset**
We use the same COLMAP loader as 3DGS and can read in data in the Colmap format and the Blender format.
The expected project structure with Colmap is
/data_dir
    /images
    /sparse
        /0
            /cameras.txt, images.txt
    lidar_pointcloud.ply
The expected project structure with the Blender format is
/data_dir
    /images
    transforms_train.json
    optional: transforms_test.json
    lidar_pointcloud.ply
We expect the filepaths in the json files to include the .jpg/.png extension

The training parameters can be found in arguments/__init__.py and may need to be finetuned to the characteristics of the LiDAR/dataset used.
Relevant parameters for tuning are self.w and self.h in the Preprocessing parameters, which should be set to the same resolution as the training images. Additionally self.curvature_threshold_preprocessing has to be finetuned such that there is a nice seperation between high curvature and low curvature areas in points3D.ply after subsampling. The parameter range is 0-1.
The filtering parameters may also need adjustment.


## Acknowledgements
This project is built upon [2DGS](https://github.com/hbb1/2d-gaussian-splatting).


## Citation
If you find our code or paper helps, please consider citing:
```bibtex
@misc{strobel2025surffill,
  author       = {Svenja Strobel and Matthias Innmann and Bernhard Egger and 
                 Marc Stamminger and Linus Franke},
  title        = {{SurfFill}: Completion of LiDAR Point Clouds via Gaussian Surfel Splatting},
  month        = {Dez},
  eprint       = {2512.03010},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  year         = {2025},
  url          = {https://lfranke.github.io/surffill}
}
```
