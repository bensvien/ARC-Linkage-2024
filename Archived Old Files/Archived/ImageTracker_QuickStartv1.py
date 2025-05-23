# -*- coding: utf-8 -*- 
"""
Created on Thu Feb  12 11:01:45 2025

@author: Dr. Benjamin Vien

Demonstration: Using MAT query points for CoTracker Visualizer.
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
from torch.cuda.amp import autocast  # if needed

# Check CUDA
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

# ------------------------------------------------------------
# Load Video and Create Video Tensor
# ------------------------------------------------------------
print("--------------------------------------------------------------")
print("Loading Video...")
video_path = './Input Files/IMG_7296.mp4'
frames = iio.imread(video_path, plugin='FFMPEG')
print("Video Loaded!")

# For example, we select frames 170 to 300 with step 2.
frames_clip = frames[170:300:2]
video_tensor = torch.tensor(frames_clip).permute(0, 3, 1, 2).float().to(device)  # [T, C, H, W]
video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension -> [1, T, C, H, W]
print("Video tensor shape:", video_tensor.shape)

# ------------------------------------------------------------
# Load MATLAB Query Points and Prepare Query Tensor
# ------------------------------------------------------------
have_qpoints = True
if have_qpoints:
    print("Loading MAT query points...")
    mat_data = sio.loadmat('./matlab_files/saved_objC_ordered_NaN_fixed.mat')
    # Assuming the MATLAB file contains variable 'data' with shape [1500, 2, Total_Frames]
    saved_objC = mat_data['data']
    saved_objC_tensor = torch.tensor(saved_objC, dtype=torch.float32).to(device)
    # Adjust coordinates (if they start at 1, subtract 1)
    saved_objC_tensor[:, :2, :] = saved_objC_tensor[:, :2, :] - 1
    print("Loaded MAT query points. Shape:", saved_objC_tensor.shape)
    
    # Select query points from a specific frame.
    # Note: This index refers to the MATLAB query points' frames.
    idx = 0  # For example, use the first frame of the MAT queries.
    query_points = saved_objC_tensor[:, :, idx]  # Shape: (1500, 2)
    
    # Add a column for frame index → [frame_index, x, y]
    frame_indices = torch.zeros((query_points.shape[0], 1), dtype=torch.float32, device=device)
    query_points_with_frame = torch.cat((frame_indices, query_points), dim=1)  # Shape: (1500, 3)
    
    # Add a batch dimension: final shape [1, 1500, 3]
    combined_tensor = query_points_with_frame.unsqueeze(0).to(device)
    print("Combined query tensor shape:", combined_tensor.shape)
else:
    combined_tensor = None

# ------------------------------------------------------------
# Load CoTracker Model from Local Repository and Run Tracker
# ------------------------------------------------------------
print("Loading CoTracker model from local repository...")
cotracker = torch.hub.load('./co-tracker', 'cotracker3_offline', source='local').to(device)
# Set model to evaluation mode for inference.
cotracker.eval()
print("Model Loaded!")

# Run the tracker using the MAT query points.
print("Running the tracker...")
with torch.no_grad():
    if combined_tensor is not None:
        pred_tracks, pred_visibility = cotracker(video_tensor, queries=combined_tensor)
    else:
        # If no query points are provided, fallback to grid points.
        pred_tracks, pred_visibility = cotracker(video_tensor, grid_size=40)
print("Tracker run complete.")
print("pred_tracks shape:", pred_tracks.shape)
print("pred_visibility shape:", pred_visibility.shape)

# ------------------------------------------------------------
# Visualize the Tracking Results with CoTracker Visualizer
# ------------------------------------------------------------
from cotracker.utils.visualizer import Visualizer
vis = Visualizer(save_dir="./saved_videos/ImageTracker/checking", pad_value=1, linewidth=3)
vis.visualize(video_tensor, pred_tracks, pred_visibility)
