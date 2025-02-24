# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:48:06 2025

@author: Dr. Benjamin Vien


Demonstration: Visualizing ground-truth query points as predicted tracks,
with user-defined start, end, and step parameters.
"""

## TO INSTALL
# !pip install torch torchvision
# !pip install imageio
# !pip install matplotlib
# !pip install imageio[ffmpeg]

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.cuda.amp import autocast  # For mixed precision if needed
import cv2

# Define device
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

use_local = False #Use False less GPU memory...
if not use_local:
    cotracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device)
    print(f"GitHub CoTracker3 is using device: {device}")
else:
    cotracker = torch.hub.load('./co-tracker', 'cotracker3_offline', source='local').to(device)
    print(f"Local CoTracker3 is using device: {device}")


def load_video_tensor(video_path, clip_start, clip_end, step, resize_dim=None):

    frames = iio.imread(video_path, plugin='FFMPEG')
    # Clip and subsample frames.
    frames_clip = frames[clip_start:clip_end:step]
    if resize_dim is not None:
        frames_clip = np.array([cv2.resize(frame, resize_dim) for frame in frames_clip])
    # Convert to tensor: initial shape [T, H, W, C]
    video_tensor = torch.tensor(frames_clip).permute(0, 3, 1, 2).float()
    video_tensor = video_tensor.unsqueeze(0).to(device)
    return video_tensor


def load_mat_query_points(mat_path, clip_start, clip_end, step):

    mat_data = sio.loadmat(mat_path)
    saved_objC = mat_data['data']  # shape: (N, 2, Total_Frames)
    saved_objC_tensor = torch.tensor(saved_objC, dtype=torch.float32).to(device)
    # Adjust coordinates if needed (MATLAB indices start at 1)
    saved_objC_tensor[:, :2, :] = saved_objC_tensor[:, :2, :] - 1
    # Clip and subsample the query points to match the video clip.
    saved_objC_tensor_clipped = saved_objC_tensor[:, :, clip_start:clip_end:step]
    # Permute dimensions: from (N, 2, T_clip) to (T_clip, N, 2)
    pred_tracks = saved_objC_tensor_clipped.permute(2, 0, 1)
    # Add batch dimension: [1, T_clip, N, 2]
    pred_tracks = pred_tracks.unsqueeze(0)
    # Create a dummy visibility tensor (all ones) of shape [1, T_clip, N, 1]
    pred_visibility = torch.ones((pred_tracks.shape[0], pred_tracks.shape[1],
                                    pred_tracks.shape[2], 1), dtype=torch.float32).to(device)
    return pred_tracks, pred_visibility

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------
def main():
    # User-defined parameters:
    clip_start = 170
    clip_end = 300
    step = 2
    resize_dim = None  # Or e.g. (640,480)
    
    video_path = "./Input Files/IMG_7296.mp4"
    mat_path = "./matlab_files/saved_objC_ordered_NaN_fixed.mat"
    
    # Load video tensor.
    video_tensor = load_video_tensor(video_path, clip_start, clip_end, step, resize_dim)
    print("Video tensor shape:", video_tensor.shape)
    
    # Load MATLAB query points and adjust them to the clip.
    have_qpoints = True
    if have_qpoints:
        pred_tracks, pred_visibility = load_mat_query_points(mat_path, clip_start, clip_end, step)
        print("Adjusted query (ground truth) pred_tracks shape:", pred_tracks.shape)
    else:
        pred_tracks, pred_visibility = None, None
    
    # Optionally, you can visualize one frame to check the video.
    # For example:
    # frame = video_tensor[0, 0].detach().cpu().permute(1,2,0).numpy()
    # plt.imshow(frame); plt.show()
    
    from cotracker.utils.visualizer import Visualizer
    vis = Visualizer(save_dir="./saved_videos/ImageTracker", pad_value=1, linewidth=3)
    if pred_tracks is not None and pred_visibility is not None:
        vis.visualize(video_tensor, pred_tracks, pred_visibility)
    else:
        print("No valid query points available for visualization.")

if __name__ == '__main__':
    main()
