# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:53:08 2025

@author: Melly
"""

# synthetic_utils.py
import cv2
import numpy as np
import torch
import imageio.v3 as iio
import scipy.io as sio
import matplotlib.pyplot as plt
import imageio
import matplotlib.cm as cm

def queryref_points(have_qpoints,have_refpoints,idx,device):
    have_qpoints = True
    have_refpoints = False

    if have_qpoints:
        # Load the MATLAB file for query points.
        mat_data = sio.loadmat('./matlab_files/saved_objC_ordered_NaN_fixed.mat')
        saved_objC = mat_data['data']
        saved_objC_tensor = torch.tensor(saved_objC, dtype=torch.float32).to(device) 
        saved_objC_tensor[:, :2, :] = saved_objC_tensor[:, :2, :] - 1
        print('Loaded MAT Query Points')
        print(type(saved_objC_tensor))
        print(saved_objC_tensor.shape)

    if have_refpoints:
        ref_points = torch.tensor([[
            [0, 500, 550],
            [0, 600, 650],
            [0, 800, 850],
            [0, 900, 850],
        ]], dtype=torch.float32).to(device) 

    combined_tensor = None
    #idx = 170  # Frame index for initializing queries

    if have_qpoints:
        query_points_0 = saved_objC_tensor[:, :, idx]  # Shape: (1500, 2)
        # Step 1: Add frame index → [0, x, y]
        frame_indices = torch.zeros((query_points_0.shape[0], 1), dtype=torch.float32, device=device)
        query_points_0_with_frame = torch.cat((frame_indices, query_points_0), dim=1)  # Shape: (1500, 3)
        print("Query Points for Frame 0:", query_points_0_with_frame.shape)
        if not have_refpoints:
            combined_tensor = query_points_0_with_frame.unsqueeze(0).to(device)  # (1, 1500, 3)

    if have_refpoints:
        ref_points_reshaped = ref_points.squeeze(0)  # Shape: (N, 3)
        print("Reference Points Shape:", ref_points_reshaped.shape)
        if not have_qpoints:
            combined_tensor = ref_points_reshaped.unsqueeze(0).to(device)  # (1, N, 3)

    if have_refpoints and have_qpoints:
        combined_tensor = torch.cat((ref_points_reshaped, query_points_0_with_frame), dim=0).to(device)  # (N+1500, 3)
        combined_tensor = combined_tensor.unsqueeze(0)  # (1, N+1500, 3)
        print("Combined Tensor Shape:", combined_tensor.shape)

    if combined_tensor is not None:
        print("Final Combined Tensor Shape:", combined_tensor.shape)
    else:
        print("No valid points available for combination.")
    if have_refpoints:
        return combined_tensor, saved_objC_tensor, ref_points
    else:
        ref_points = None
        return combined_tensor, saved_objC_tensor, ref_points

def checkframes(check_frame,frames,idx,have_qpoints,have_refpoints,saved_objC_tensor,ref_points=None):
    if check_frame:
        #idx = 400
        H, W, _ = frames[idx].shape
        plt.figure(dpi=600)
        plt.imshow(frames[idx], extent=[0, W, H, 0], interpolation='none')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis("off")
        if have_qpoints:
            query_points = saved_objC_tensor[:, :, idx].cpu().numpy()
            plt.scatter(query_points[:, 0], query_points[:, 1], color='red', marker='.', label='Query Points', s=.5)
        if have_refpoints:
            ref_points_np = ref_points.cpu().numpy().squeeze(0)
            plt.scatter(ref_points_np[:, 1], ref_points_np[:, 2], color='green', marker='+', alpha=0.3, s=40, label="All Ref Points")
            ref_points_frame = ref_points_np[ref_points_np[:, 0] == idx]
            if ref_points_frame.size > 0:
                plt.scatter(ref_points_frame[:, 1], ref_points_frame[:, 2], color='blue', marker='+', label='Active Ref Points', s=40)
        plt.legend()
        plt.show()
        
        
def processingframe(frames,start0,end0,step0,combined_tensor,device):
    print("Creating Video Tensor...")
    #frames = frames[170:480:60]  # Use frames 170 to 300 with step 2
    frames = frames[start0:end0:step0]
    video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # Shape: [1, T, C, H, W]
    del frames
    print("Completed Video Tensor!")

    # Check and clean combined_tensor if needed.
    if combined_tensor is not None:
        print("Before NaN removal:", combined_tensor.shape)
        if torch.isnan(combined_tensor).any():
            nan_mask = torch.isnan(combined_tensor).any(dim=2)
            combined_tensor = combined_tensor[:, ~nan_mask.squeeze(0), :]
        else:
            print('No NaN')
        print("After NaN removal:", combined_tensor.shape)
    else:
        print("combined_tensor is empty or not initialized.")
        
    if combined_tensor is None:
        grid_size = 40  # Use grid points if no query points available
    return combined_tensor,video_tensor

def plot_single_frame_with_gradient(video_tensor, tracks, frame_idx):
    """
    Plots a single video frame (extracted from video_tensor) with overlaid tracking points in gradient colors.
    
    Parameters:
      video_tensor: A torch tensor of shape [B, T, C, H, W].
      tracks: A NumPy array or tensor of shape [T, N, 2] containing the tracked point coordinates.
      frame_idx: The index of the frame to visualize.
    """
    # Extract the frame from the video tensor.
    frame = video_tensor[0, frame_idx].detach().cpu().permute(1, 2, 0).numpy()
    #frame = video_tensor[frame_idx,0].detach().cpu().permute(1, 2, 0).numpy()
    # Normalize if the frame values are in [0, 255]
    print("Frame min:", frame.min(), "Frame max:", frame.max())
    if frame.max() > 1:
        frame = frame / 255.0
    
    print("Frame min:", frame.min(), "Frame max:", frame.max())
    
    plt.figure(figsize=(10, 8))
    plt.imshow(frame, vmin=0, vmax=1)
    
    num_points = tracks.shape[1]
    #colors = cm.jet_r(np.linspace(0, 1, num_points))
    #colors = cm.get_cmap("gist_rainbow")
    colors = cm.get_cmap("gist_rainbow")(np.linspace(0, 1, num_points))
    
    for point_idx in range(num_points):
        x, y = tracks[frame_idx, point_idx]
        plt.scatter(x, y, color=colors[point_idx], s=4*2, edgecolors='None', linewidth=4)
    
    plt.title(f"Tracking Visualization - Frame {frame_idx}")
    plt.axis("off")
    plt.show()

def plot_single_frame_with_gradient_visibility(video_tensor, tracks, visibility, frame_idx):
    """
    Plots a single video frame (extracted from video_tensor) with tracking points,
    using gradient colors and weighting the point display by their visibility.
    
    Parameters:
      video_tensor: A torch tensor of shape [B, T, C, H, W].
      tracks: A NumPy array or tensor of shape [T, N, 2] containing the tracked coordinates.
      visibility: A torch tensor of shape [B, T, N] containing visibility scores.
      frame_idx: The index of the frame to visualize.
    """
    # Extract the frame.
    frame = video_tensor[0, frame_idx].detach().cpu().permute(1, 2, 0).numpy()
    if frame.max() > 1:
        frame = frame / 255.0
    
    plt.figure(figsize=(10, 8))
    plt.imshow(frame, vmin=0, vmax=1)
    
    num_points = tracks.shape[1]
    colors = cm.jet_r(np.linspace(0, 1, num_points))
    
    visibility_np = visibility.squeeze(0).cpu().numpy()  # Shape: [T, N]
    visibilities = visibility_np[frame_idx, :].astype(np.float32)
    visibilities = (visibilities - visibilities.min()) / (visibilities.max() - visibilities.min() + 1e-6)
    
    for point_idx in range(num_points):
        x, y = tracks[frame_idx, point_idx]
        alpha = visibilities[point_idx]
        size = 5 + 20 * alpha
        plt.scatter(x, y, color=colors[point_idx], s=size, alpha=alpha, edgecolors='None', linewidth=0.3)
    
    plt.title(f"Tracking Visualization - Frame {frame_idx} (Visibility Weighted)")
    plt.axis("off")
    plt.show()

def extend_tensor_along_time(tensor, extra,dim0):
    # Each frame is repeated (extra+1) times along dimension 1 (time dimension)
    return tensor.repeat_interleave(extra + 1, dim=dim0)

def save_video_tensor(video_tensor, save_path, fps=30, extra=0):
    """
    Save a video from a video tensor to a file.

    Args:
        video_tensor (torch.Tensor): Tensor of shape (B, T, C, H, W) with pixel values in [0, 1] or uint8.
        save_path (str): Path to save the video (e.g., "output.mp4").
        fps (int): Frames per second for the output video.
        extra (int): Number of extra copies to add for each frame.
                     For example, if extra=5, each frame is repeated 6 times (original + 5 extra).
    """
    # Ensure we work with a single batch element
    if video_tensor.dim() != 5:
        raise ValueError("video_tensor should be of shape (B, T, C, H, W)")
    
    # Select the first element in the batch
    video_tensor = video_tensor[0]  # Now shape is (T, C, H, W)
    
    # Optionally, extend the video by repeating each frame
    if extra > 0:
        video_tensor = video_tensor.repeat_interleave(extra + 1, dim=0)  # Extend along time dimension
    
    # Convert tensor to numpy array
    # If the tensor is in float format, we assume values are in [0, 1] and convert to uint8.
    video_np = video_tensor.cpu().numpy()
    if video_np.dtype != np.uint8:
        video_np = (video_np).astype(np.uint8)
    
    # Rearrange to (T, H, W, C) for imageio
    video_np = video_np.transpose(0, 2, 3, 1)
    
    # Swap channels if the colors appear wrong (e.g., blue tint)
    # This converts from RGB to BGR or vice versa.
    #video_np = video_np[..., [2, 1, 0]]
    
    # Write the video
    writer = imageio.get_writer(save_path, fps=fps)
    for frame in video_np:
        writer.append_data(frame)
    writer.close()
    
    print(f"Video saved to {save_path}")

def plot_single_frame_displacement(video_tensor, tracks, frame_idx, disp_type="total"):
    """
    Plots a single video frame (extracted from video_tensor) with overlaid tracking points in gradient colors.
    
    Parameters:
      video_tensor: A torch tensor of shape [B, T, C, H, W].
      tracks: A NumPy array or tensor of shape [T, N, 2] containing the tracked point coordinates.
      frame_idx: The index of the frame to visualize.
      disp_type: Type of displacement to visualize ('horizontal', 'vertical', or 'total').
    """
    # Extract the frame from the video tensor
    frame = video_tensor[0, frame_idx].detach().cpu().permute(1, 2, 0).numpy()
    
    # Normalize if the frame values are in [0, 255]
    if frame.max() > 1:
        frame = frame / 255.0
    
    plt.figure(figsize=(10, 8))
    plt.imshow(frame, vmin=0, vmax=1)
    
    num_points = tracks.shape[1]
    colors = cm.get_cmap("gist_rainbow")(np.linspace(0, 1, num_points))
    
    # Compute displacements relative to the first frame
    ref_positions = tracks[0]  # Initial positions
    curr_positions = tracks[frame_idx]  # Current positions
    
    displacements = curr_positions - ref_positions
    
    if disp_type == "horizontal":
        magnitudes = displacements[:, 0]  # X displacement
    elif disp_type == "vertical":
        magnitudes = displacements[:, 1]  # Y displacement
    else:
        magnitudes = np.linalg.norm(displacements, axis=1)  # Total displacement
    
    norm = plt.Normalize(vmin=np.min(magnitudes), vmax=np.max(magnitudes))
    
    for point_idx in range(num_points):
        x, y = curr_positions[point_idx]
        plt.scatter(x, y, color=cm.jet(norm(magnitudes[point_idx])), s=6, edgecolors='None', linewidth=3)
    
    plt.title(f"Tracking Visualization - Frame {frame_idx} ({disp_type} displacement)")
    plt.axis("off")
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.jet), label=f"{disp_type.capitalize()} displacement")
    plt.show()
    
from scipy.interpolate import griddata

from scipy.spatial import Delaunay

def plot_single_frame_displacement_map(video_tensor, tracks, frame_idx, disp_type="total",caxis_limits=None):
    """
    Plots a single video frame (extracted from video_tensor) with a smooth 2D displacement map.
    
    Parameters:
      video_tensor: A torch tensor of shape [B, T, C, H, W].
      tracks: A NumPy array or tensor of shape [T, N, 2] containing the tracked point coordinates.
      frame_idx: The index of the frame to visualize.
      disp_type: Type of displacement to visualize ('horizontal', 'vertical', or 'total').
    """
    # Extract the frame from the video tensor
    # Extract the frame from the video tensor
    frame = video_tensor[0, frame_idx].detach().cpu().permute(1, 2, 0).numpy()
    height, width = frame.shape[:2]
    
    # Normalize if the frame values are in [0, 255]
    if frame.max() > 1:
        frame = frame / 255.0
    
    plt.figure(figsize=(10, 8))
    plt.imshow(frame, vmin=0, vmax=1)
    
    num_points = tracks.shape[1]
    
    # Compute displacements relative to the first frame
    ref_positions = tracks[0]  # Initial positions
    curr_positions = tracks[frame_idx]  # Current positions
    
    displacements = curr_positions - ref_positions
    
    if disp_type == "horizontal":
        magnitudes = displacements[:, 0]  # X displacement
    elif disp_type == "vertical":
        magnitudes = displacements[:, 1]  # Y displacement
    else:
        magnitudes = np.linalg.norm(displacements, axis=1)  # Total displacement
    
    # Generate a full-frame grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, width - 1, 200), np.linspace(0, height - 1, 200)
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Create a mask for interpolation within the convex hull
    hull = Delaunay(curr_positions)
    mask = hull.find_simplex(grid_points) >= 0
    
    # Interpolate values constrained by the convex hull
    grid_z = griddata(curr_positions, magnitudes, grid_points[mask], method='cubic', fill_value=np.nan)
    full_grid_z = np.full(grid_x.shape, np.nan)
    full_grid_z.ravel()[mask] = grid_z
    
    img=plt.imshow(full_grid_z, extent=(0, width, height, 0), origin='upper', cmap=cm.jet, alpha=0.6)
    plt.colorbar(label=f"{disp_type.capitalize()} displacement")
    if caxis_limits:
        img.set_clim(*caxis_limits)
    
    plt.title(f"Tracking Visualization - Frame {frame_idx} ({disp_type} displacement)")
    plt.axis("off")
    plt.show()


# Function to generate random points within the quadrilateral
# Function to generate random points within the quadrilateral
def random_points_in_quad(corners, num_points=1024,device='cuda'):
    # Ensure corners is a float tensor
    corners = corners.to(torch.float32)
    device='cuda'
    # Triangulate the quadrilateral
    tri1 = corners[[0, 1, 2]]
    tri2 = corners[[0, 2, 3]]

    # Function to generate random points within a triangle
    def random_points_in_triangle(triangle, num_points,device):
        u = torch.sqrt(torch.rand(num_points, 1))
        v = torch.rand(num_points, 1)

        barycentric = torch.cat([
            1 - u,
            u * (1 - v),
            u * v
        ], dim=1)

        return torch.einsum('ij,nj->ni', triangle, barycentric)

    # Generate points for both triangles
    points1 = random_points_in_triangle(tri1.T, num_points // 2,device)
    points2 = random_points_in_triangle(tri2.T, num_points - num_points // 2,device)

    # Combine points from both triangles
    points = torch.cat([points1, points2], dim=0)

    # Add the 0 in the first dimension
    points = torch.cat([torch.zeros(num_points, 1), points], dim=1)

    return points.unsqueeze(0).to(device) 


def random_points_in_quad_edge(corners, num_points=1024, edge_points=32,device='cuda'):
    # Ensure corners is a float tensor
    corners = corners.to(torch.float32)

    # Generate edge points uniformly
    edges = []
    for i in range(4):
        start, end = corners[i], corners[(i + 1) % 4]
        t = torch.linspace(0, 1, edge_points).unsqueeze(1)
        edge_points_tensor = (1 - t) * start + t * end
        edges.append(edge_points_tensor)

    edge_points_tensor = torch.cat(edges, dim=0)

    # Number of random points inside the quadrilateral
    remaining_points = num_points - 4 * edge_points

    # Triangulate the quadrilateral
    tri1 = corners[[0, 1, 2]]
    tri2 = corners[[0, 2, 3]]

    # Function to generate random points within a triangle
    def random_points_in_triangle(triangle, num_points):
        u = torch.sqrt(torch.rand(num_points, 1))
        v = torch.rand(num_points, 1)

        barycentric = torch.cat([
            1 - u,
            u * (1 - v),
            u * v
        ], dim=1)

        return torch.einsum('ij,nj->ni', triangle, barycentric)

    # Generate points for both triangles
    points1 = random_points_in_triangle(tri1.T, remaining_points // 2)
    points2 = random_points_in_triangle(tri2.T, remaining_points - remaining_points // 2)

    # Combine edge points and random points
    all_points = torch.cat([edge_points_tensor, points1, points2], dim=0)

    # Add the 0 in the first dimension
    all_points = torch.cat([torch.zeros(num_points, 1), all_points], dim=1)

    return all_points.unsqueeze(0).to(device)

    
def plot_single_frame_displacement_map_MATLAB(video_tensor, tracks, displacement_u, displacement_v, frame_idx, disp_type="total", caxis_limits=None):
    """
    Plots a single video frame with a 2D interpolated displacement map using precomputed displacements.

    Parameters:
        video_tensor: A torch tensor of shape [B, T, C, H, W].
        tracks: A NumPy array or tensor of shape [T, N, 2] containing tracked point coordinates.
        displacement_u: Horizontal displacements, shape [T, N, 1].
        displacement_v: Vertical displacements, shape [T, N, 1].
        frame_idx: The index of the frame to visualize.
        disp_type: One of 'horizontal', 'vertical', or 'total'.
        caxis_limits: Tuple (vmin, vmax) for color scaling.
    """
    # Prepare the frame
    frame = video_tensor[0, frame_idx].detach().cpu().permute(1, 2, 0).numpy()
    height, width = frame.shape[:2]
    if frame.max() > 1:
        frame = frame / 255.0
    
    plt.figure(figsize=(10, 8))
    plt.imshow(frame, vmin=0, vmax=1)
    
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.detach().cpu().numpy()
    
    curr_positions = tracks[frame_idx]  # [N, 2]
    
    # Extract displacements for current frame
    u = displacement_u[frame_idx, :]
    v = displacement_v[frame_idx, :]
    
    if disp_type == "horizontal":
        magnitudes = u
    elif disp_type == "vertical":
        magnitudes = v
    else:  # total
        magnitudes = np.sqrt(u**2 + v**2)
    
    # Generate interpolation grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, width - 1, 200), np.linspace(0, height - 1, 200)
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Convex hull mask
    hull = Delaunay(curr_positions)
    mask = hull.find_simplex(grid_points) >= 0

    # Interpolation
    grid_z = griddata(curr_positions, magnitudes, grid_points[mask], method='cubic', fill_value=np.nan)
    full_grid_z = np.full(grid_x.shape, np.nan)
    full_grid_z.ravel()[mask] = grid_z

    # Plot interpolated displacement
    img = plt.imshow(full_grid_z, extent=(0, width, height, 0), origin='upper', cmap=cm.jet, alpha=0.6)
    plt.colorbar(label=f"{disp_type.capitalize()} displacement")
    
    if caxis_limits:
        img.set_clim(*caxis_limits)

    plt.title(f"Displacement Map - Frame {frame_idx} ({disp_type})")
    plt.axis("off")
    plt.show()