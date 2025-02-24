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

#%% Monkey Patching
def get_new_forward(model):
    orig_forward = model.forward

    def new_forward(self, x, queries=None):
        with torch.enable_grad():
            out = orig_forward(x, queries=queries)
        return out

    return new_forward.__get__(model, type(model))


    model = torch.hub.load(
        './co-tracker', #Local Drive
        'cotracker3_offline',
        source='local'
    ).to(device)
    print("Model Loaded!")
    
    model.train()
    for param in model.parameters():
        param.requires_grad = True

model.forward = get_new_forward(model)

def train_finetune_updated(model, dataloader, optimizer, scheduler, device, num_epochs=10, 
                           use_autocast=True, use_gradscaler=False,
                           freeze_confidence_branch=True,
                           occlusion_discount=1,  # Discount factor for occluded points
                           gamma=1.1):              # Exponential weighting factor per timestep
    model.train()
    # Use Huber loss with delta=6.0 and no reduction so that we can apply custom weighting.
    criterion = nn.HuberLoss(delta=6.0, reduction='none')
    scaler = GradScaler() if use_gradscaler else None
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"\nEpoch: {epoch+1}")
        for batch in dataloader:
            video = batch['video'].to(device)        # [B, T, C, H, W]
            gt_tracks = batch['gt_tracks'].to(device)  # [B, T, N, 2]
            queries = batch['queries'].to(device)      # Expected shape: [B, N, 3]
            if queries.dim() == 2:
                queries = queries.unsqueeze(0)
            
            optimizer.zero_grad()
            if use_autocast:
                with autocast():
                    pred_tracks, pred_visibility = model(video, queries=queries)
                    if freeze_confidence_branch:
                        pred_visibility = pred_visibility.detach()
                    
                    loss_all = criterion(pred_tracks, gt_tracks)
                    
                    # If a ground truth visibility mask exists in batch, apply occlusion discount.
                    if 'gt_visibility' in batch:
                        gt_visibility = batch['gt_visibility'].to(device)
                        visibility = gt_visibility.unsqueeze(-1)
                        discount = visibility + occlusion_discount * (1 - visibility)
                        loss_all = loss_all * discount
                    
                    # Average loss over coordinate and track dimensions to get loss per timestep.
                    loss_per_timestep = loss_all.mean(dim=(-1, -2))
                    T = loss_per_timestep.shape[1]
                    # Create exponential weights per timestep.
                    weights = torch.tensor([gamma ** t for t in range(T)],
                                             device=loss_per_timestep.device,
                                             dtype=loss_per_timestep.dtype)
                    loss_weighted = loss_per_timestep * weights
                    loss = loss_weighted.mean()
                    
                    print(f"  [Autocast] pred_tracks.dtype: {pred_tracks.dtype}")
            else:
                pred_tracks, pred_visibility = model(video, queries=queries)
                if freeze_confidence_branch:
                    pred_visibility = pred_visibility.detach()
                
                loss_all = criterion(pred_tracks, gt_tracks)
                if 'gt_visibility' in batch:
                    gt_visibility = batch['gt_visibility'].to(device)
                    visibility = gt_visibility.unsqueeze(-1)
                    discount = visibility + occlusion_discount * (1 - visibility)
                    loss_all = loss_all * discount
                
                loss_per_timestep = loss_all.mean(dim=(-1, -2))
                T = loss_per_timestep.shape[1]
                weights = torch.tensor([gamma ** t for t in range(T)],
                                       device=loss_per_timestep.device,
                                       dtype=loss_per_timestep.dtype)
                loss_weighted = loss_per_timestep * weights
                loss = loss_weighted.mean()
                
                print(f"  [No Autocast] pred_tracks.dtype: {pred_tracks.dtype}")
            
            if use_gradscaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            
            scheduler.step()
            running_loss += loss.item() * video.size(0)
            torch.cuda.empty_cache()  # Optional: free unused memory
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return model
