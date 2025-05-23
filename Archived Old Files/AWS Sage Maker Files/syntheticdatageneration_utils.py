# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:24:37 2025

@author: Dr. Benjamin Vien
"""

import cv2
import numpy as np
import torch
import imageio.v3 as iio
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Synthetic Data Generation Functions
# -----------------------------
def random_affine_params(rotation_range=(-30, 30), 
                           scale_range=(0.8, 1.5), 
                           shear_x_range=(-20, 20), 
                           shear_y_range=(-20, 20),
                           translation_range=(-20, 20),
                           anisotropic=True,
                           scale_range_y=(0.8, 1.5)):
    angle = np.random.uniform(*rotation_range)
    if anisotropic:
        scale_x = np.random.uniform(*scale_range)
        scale_y = np.random.uniform(*scale_range_y)
    else:
        scale_x = scale_y = np.random.uniform(*scale_range)
    shear_x = np.random.uniform(*shear_x_range)
    shear_y = np.random.uniform(*shear_y_range)
    tx = np.random.uniform(*translation_range)
    ty = np.random.uniform(*translation_range)
    return angle, scale_x, scale_y, shear_x, shear_y, tx, ty

def get_affine_matrix(angle, scale_x, scale_y, shear_x, shear_y, tx, ty, center):
    angle_rad = np.deg2rad(angle)
    shear_x_rad = np.deg2rad(shear_x)
    shear_y_rad = np.deg2rad(shear_y)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    S = np.array([
        [scale_x, 0],
        [0, scale_y]
    ])
    Sh = np.array([
        [1, np.tan(shear_x_rad)],
        [np.tan(shear_y_rad), 1]
    ])
    M = R @ Sh @ S
    center = np.array(center)
    offset = center - M @ center + np.array([tx, ty])
    affine_matrix = np.hstack([M, offset.reshape(2,1)])
    return affine_matrix

def apply_affine(image, query_points, M):
    transformed_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                        borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((query_points.shape[0], 1))
    points_hom = np.hstack([query_points, ones])
    transformed_points = (M @ points_hom.T).T
    return transformed_image, transformed_points

def generate_interpolated_video(image, query_points, num_frames=5,
                                rotation_range=(-30,30), 
                                scale_range=(0.8,1.5),
                                shear_x_range=(-20,20), 
                                shear_y_range=(-20,20),
                                translation_range=(-20,20),
                                anisotropic=False,
                                scale_range_y=(0.8,1.5)):
    angle, scale_x, scale_y, shear_x, shear_y, tx, ty = random_affine_params(
        rotation_range, scale_range, shear_x_range, shear_y_range, translation_range,
        anisotropic, scale_range_y)
    center = (image.shape[1] / 2, image.shape[0] / 2)
    target_M = get_affine_matrix(angle, scale_x, scale_y, shear_x, shear_y, tx, ty, center)
    identity_M = np.hstack([np.eye(2), np.zeros((2,1))])
    
    video_frames = []
    query_points_list = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        M_interp = identity_M + alpha * (target_M - identity_M)
        frame_interp, query_interp = apply_affine(image, query_points, M_interp)
        video_frames.append(frame_interp)
        query_points_list.append(query_interp)
    
    return video_frames, query_points_list

def save_video(frames, output_path, fps=5):
    import imageio
    imageio.mimwrite(output_path, frames, fps=fps)
    print("Video saved to", output_path)

def save_video_with_overlay(frames, query_points_list, output_path, fps=5,
                            point_color=(255,0,0), point_radius=3):
    overlay_frames = []
    for frame, points in zip(frames, query_points_list):
        frame_overlay = frame.copy()
        for (x, y) in points:
            cv2.circle(frame_overlay, (int(x), int(y)), point_radius, point_color, -1)
        overlay_frames.append(frame_overlay)
    save_video(overlay_frames, output_path, fps=fps)

# -----------------------------
# Synthetic Dataset for Fine-Tuning
# -----------------------------
class SyntheticDataset(Dataset):
    def __init__(self, base_image, base_query_points, num_samples=20, num_frames=5, transform_params=None):
        self.samples = []
        self.base_query_points = base_query_points
        if transform_params is None:
            transform_params = {
                'rotation_range': (-45, 45),
                'scale_range': (0.8, 1.5),
                'shear_x_range': (-40, 40),
                'shear_y_range': (-40, 40),
                'translation_range': (-50, 50),
                'anisotropic': True,
                'scale_range_y': (0.8, 1.5)
            }
        for i in range(num_samples):
            video_frames, query_points_list = generate_interpolated_video(
                base_image, base_query_points, num_frames=num_frames,
                rotation_range=transform_params['rotation_range'],
                scale_range=transform_params['scale_range'],
                shear_x_range=transform_params['shear_x_range'],
                shear_y_range=transform_params['shear_y_range'],
                translation_range=transform_params['translation_range'],
                anisotropic=transform_params['anisotropic'],
                scale_range_y=transform_params['scale_range_y']
            )
            self.samples.append((video_frames, query_points_list))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_frames, query_points_list = self.samples[idx]
        # Return video as tensor with shape [T, C, H, W] (no batch dimension)
        
        #old
        # video_tensor = torch.tensor(video_frames).permute(0, 3, 1, 2).float() / 255.0
        
        #new
        video_np = np.array(video_frames)  # Convert the list to a single NumPy array
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
        
        
        # Return pred_tracks with shape [T, N, 2]
        pred_tracks = torch.tensor(query_points_list, dtype=torch.float32)
        # Dummy visibility: [T, N, 1]
        pred_visibility = torch.ones((pred_tracks.shape[0], pred_tracks.shape[1], 1), dtype=torch.float32)
        # Create queries from base query points (add a column of zeros)
        base_q = self.base_query_points  # shape (N,2)
        zeros = np.zeros((base_q.shape[0], 1), dtype=np.float32)
        queries_np = np.hstack([zeros, base_q])  # shape (N,3)
        queries = torch.tensor(queries_np, dtype=torch.float32)
        return {'video': video_tensor, 'pred_tracks': pred_tracks, 
                'pred_visibility': pred_visibility, 'queries': queries}
# -----------------------------
# Training Function for Fine-Tuning
# -----------------------------
def train_finetune_updated(model, dataloader, optimizer, scheduler, device, num_epochs=10, 
                           use_autocast=True, use_gradscaler=True,
                           freeze_confidence_branch=True,
                           occlusion_discount=1,
                           gamma=1.1,
                           early_stop_patience=10, 
                           early_stop_min_delta=1e-4):
    model.train()
    criterion = torch.nn.HuberLoss(delta=6.0, reduction='none')
    scaler = GradScaler() if use_gradscaler else None
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"\nEpoch: {epoch+1}")
        batch_counter = 0
        for batch in dataloader:
            batch_counter += 1
            # Expected shapes after collation:
            # video: [B, T, C, H, W]; pred_tracks: [B, T, N, 2]; queries: [B, N, 3]
            video = batch['video'].to(device)
            gt_tracks = batch['pred_tracks'].to(device)
            queries = batch['queries'].to(device)
            if queries.dim() == 2:
                queries = queries.unsqueeze(0)
            optimizer.zero_grad()
            if use_autocast:
                with autocast():
                    pred_tracks, pred_visibility = model(video, queries=queries)
                    if freeze_confidence_branch:
                        pred_visibility = pred_visibility.detach()
                    loss_all = criterion(pred_tracks, gt_tracks)
                    loss_per_timestep = loss_all.mean(dim=(-1, -2))
                    T = loss_per_timestep.shape[1]
                    weights = torch.tensor([gamma ** t for t in range(T)],
                                             device=loss_per_timestep.device,
                                             dtype=loss_per_timestep.dtype)
                    loss = (loss_per_timestep * weights).mean()
                    print(f"  [Autocast] [Batch {batch_counter}] pred_tracks.dtype: {pred_tracks.dtype}")
            else:
                pred_tracks, pred_visibility = model(video, queries=queries)
                if freeze_confidence_branch:
                    pred_visibility = pred_visibility.detach()
                loss_all = criterion(pred_tracks, gt_tracks)
                loss_per_timestep = loss_all.mean(dim=(-1, -2))
                T = loss_per_timestep.shape[1]
                weights = torch.tensor([gamma ** t for t in range(T)],
                                       device=loss_per_timestep.device,
                                       dtype=loss_per_timestep.dtype)
                loss = (loss_per_timestep * weights).mean()
                print(f"  [No Autocast] [Batch {batch_counter}] pred_tracks.dtype: {pred_tracks.dtype}")
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
            
            # Execute scheduler step only if scheduler is provided.
            if scheduler is not None:
                scheduler.step()
            
            running_loss += loss.item() * video.size(0)
            torch.cuda.empty_cache()
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss - early_stop_min_delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered!")
                break
    return model

def fetch_optimizer(model, lr=0.0005, weight_decay=0.00001, num_steps=200000, use_scheduler=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            lr,
            num_steps + 100,
            pct_start=0.0,
            cycle_momentum=False,
            anneal_strategy="cos",
        )
    else:
        scheduler = None
    return optimizer, scheduler

def print_model_summary(model):
    children = list(model.named_children())
    print(f"Total number of top-level layers: {len(children)}")
    for i, (name, _) in enumerate(children):
        print(f"Layer {i}: {name}")