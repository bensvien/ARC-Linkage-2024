# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:00:21 2025
Updated 6/2/2025
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

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
# 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

#%% Call CoTracker3 Offline Mode
cotracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device)
print(f"CoTracker3 is using device: {device}")
