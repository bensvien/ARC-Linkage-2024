# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:25:18 2025

@author: Melly
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
import numpy as np
import scipy.io
import cv2  # for optional resizing
# 
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
torch.cuda.empty_cache()
