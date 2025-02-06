# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:00:21 2025

@author: Dr. Benjamin Vien

Demonstration:
"""

# !pip install torch torchvision
# !pip install imageio
# !pip install matplotlib
# !pip install imageio[ffmpeg]

import os
# Set the environment variable 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Libraries
import torch
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import time
import scipy.io as sio

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
# 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()




#%% Call Offline Mode
cotracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device)
print(f"CoTracker3 is using device: {device}")

#Have Queries Points
have_qpoints='True'
have_refpoints='True'

if have_qpoints:
# Load the MATLAB file
    mat_data = sio.loadmat('./matlab_files/saved_objC.mat')
    saved_objC = mat_data['saved_objC']
    saved_objC_tensor = torch.tensor(saved_objC, dtype=torch.float32).to(device) 
    saved_objC_tensor[:, :2, :] = saved_objC_tensor[:, :2, :] - 1
    
    print('Loaded MAT Queries Points')
    print(type(saved_objC))
    print(saved_objC.shape)
    
if have_refpoints:
    ref_points = torch.tensor([[
        [0, 500, 550],  # For example, point 1: frame 0, x=100, y=150
        [0, 600, 650],  # For example, point 2: frame 0, x=200, y=250
        [0, 800, 850],  # For example, point 3: frame 0, x=300, y=350
    ]], dtype=torch.float32).to(device) 


video_path = './Input Files/IMG_7296.mp4'
frames = iio.imread(video_path, plugin='FFMPEG')  # Read video frames FFMPEG must be capitalised

#%% Check Frame Image


idx = 0

H, W, _ = frames[idx].shape

plt.figure(dpi=600)
plt.imshow(frames[idx], extent=[0, W, H, 0], interpolation='none')
plt.gca().set_aspect('equal', adjustable='box')  # Ensures the aspect ratio is maintained
# plt.title(f"Frame {idx}")
plt.axis("off")

# Plot query points in cyan circles
if have_qpoints:
    query_points = saved_objC_tensor[:, :, idx].cpu().numpy()
    plt.scatter(query_points[:, 0], query_points[:, 1],
                color='red', marker='.', label='Query Points', s=.5)

if have_refpoints:

    ref_points_np = ref_points.cpu().numpy().squeeze(0)  # Now shape is (N, 3)
    ref_points_frame = ref_points_np[ref_points_np[:, 0] == idx]
    
    if ref_points_frame.size > 0:
        plt.scatter(ref_points_frame[:, 1], ref_points_frame[:, 2],
                    color='blue', marker='+', label='Reference Points', s=40)

# Add legend and show the plot
plt.legend()
plt.show()
