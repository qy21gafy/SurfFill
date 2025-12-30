import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn.functional as F
import time
from preprocess.uncertainty.data.dataloader_custom import CustomLoader
from preprocess.uncertainty.models.NNET import NNET
import preprocess.uncertainty.utils.utils as utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test(model, test_loader, device, results_dir):
    alpha_max = 60
    kappa_max = 30

    total_time = 0.0  
    num_iterations = len(test_loader)  

    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            start_time = time.time()
            img = data_dict['img'].to(device)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()

            # save results
            img_name = os.path.basename(data_dict['img_name'][0])
            #print(img_name)
            # 1. save input image
            img = utils.unnormalize(img[0, ...])
            #print(results_dir)
            target_path = '%s/%s_img.png' % (results_dir, img_name)
            #plt.imsave(target_path, img)

            # 2. predicted normal
            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)                  # (B, H, W, 3)

            target_path = '%s/%s_pred_norm.png' % (results_dir, img_name)
            #plt.imsave(target_path, pred_norm_rgb[0, :, :, :])


            # 4. predicted uncertainty
            pred_alpha = utils.kappa_to_alpha(pred_kappa)
            target_path = '%s/%s_pred_alpha.png' % (results_dir, img_name)
            plt.imsave(target_path, pred_alpha[0, :, :, 0], vmin=0.0, vmax=alpha_max, cmap='gray')

            end_time = time.time()  
            iteration_time = end_time - start_time  
            total_time += iteration_time  

    # Calculate average runtime per iteration
    average_time = total_time / num_iterations if num_iterations > 0 else 0

    # Print the total and average runtime
    print(f"Total Runtime: {total_time:.4f} seconds")
    print(f"Average Runtime per Iteration: {average_time:.4f} seconds")



def generate_uncertainty_maps(
    imgs_dir,
    input_height=1024,
    input_width=1024,
    architecture='BN',
    pretrained='scannet',
    sampling_ratio=0.4,
    importance_ratio=0.4,
):
    Params = namedtuple('Params', [
    'architecture',
    'pretrained',
    'sampling_ratio',
    'importance_ratio',
    'input_height',
    'input_width',
    'imgs_dir'
    ])

    # Create an instance of Params
    params = Params(
        architecture=architecture,
        pretrained=pretrained,
        sampling_ratio=sampling_ratio,
        importance_ratio=importance_ratio,
        input_height=input_height,
        input_width=input_width,
        imgs_dir=imgs_dir
    )

    # Set device
    device = torch.device('cuda:0')

    # Load checkpoint
    checkpoint = f'./checkpoints/{pretrained}.pt'
    print(f'Loading checkpoint... {checkpoint}')
    model = NNET(params).to(device)
    model = utils.load_checkpoint(checkpoint, model)
    model.eval()
    print('Loading checkpoint... / done')

    # Prepare test loader
    results_dir = os.path.join(os.path.dirname(imgs_dir), "uncertainty")
    os.makedirs(results_dir, exist_ok=True)
    test_loader = CustomLoader(params, params.imgs_dir).data

    # Run test
    test(model, test_loader, device, results_dir)

