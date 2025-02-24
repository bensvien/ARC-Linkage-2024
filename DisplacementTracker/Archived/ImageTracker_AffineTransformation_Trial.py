"""
Created on Fri Feb 14 12:29:00 2025


@author: Dr. Benjamin Vien

Demonstration: Generating synthetic augmented video for fine-tuning,
with query point trajectories, and saving the video with overlaid points.
Now with an option for anisotropic scaling.
"""

## TO INSTALL
# !pip install torch torchvision
# !pip install imageio
# !pip install matplotlib
# !pip install imageio[ffmpeg]
# !pip install opencv-python

import cv2
import numpy as np
import torch
import imageio.v3 as iio
import scipy.io as sio
import matplotlib.pyplot as plt

def random_affine_params(rotation_range=(-30, 30), 
                           scale_range=(0.5, 1.5), 
                           shear_range=(-20, 20), 
                           translation_range=(-50, 50),
                           anisotropic=False,
                           scale_range_y=None):
    """
    Generate random affine transformation parameters.
    
    Parameters:
      rotation_range: Tuple of min and max rotation in degrees.
      scale_range: Tuple of min and max scaling for x (if anisotropic is False, used for both axes).
      shear_range: Tuple for shear in degrees.
      translation_range: Tuple for translation in pixels (applied in both x and y).
      anisotropic: Boolean flag. If True, generate independent scale for x and y.
      scale_range_y: Tuple for y-scale if anisotropic is True. If None, use scale_range.
      
    Returns:
      angle (degrees), scale_x, scale_y, shear (degrees), tx, ty.
    """
    angle = np.random.uniform(*rotation_range)
    if anisotropic:
        scale_x = np.random.uniform(*scale_range)
        if scale_range_y is None:
            scale_range_y = scale_range
        scale_y = np.random.uniform(*scale_range_y)
    else:
        scale_x = scale_y = np.random.uniform(*scale_range)
    shear = np.random.uniform(*shear_range)
    tx = np.random.uniform(*translation_range)
    ty = np.random.uniform(*translation_range)
    return angle, scale_x, scale_y, shear, tx, ty

def get_affine_matrix(angle, scale_x, scale_y, shear, tx, ty, center):
    """
    Construct a 2x3 affine transformation matrix.
    Combines rotation, anisotropic scaling, shearing (x-direction), and translation about the center.
    """
    angle_rad = np.deg2rad(angle)
    shear_rad = np.deg2rad(shear)
    
    # Rotation matrix.
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    # Anisotropic scaling matrix.
    S = np.array([
        [scale_x, 0],
        [0, scale_y]
    ])
    # Shear matrix (shear in x direction).
    Sh = np.array([
        [1, np.tan(shear_rad)],
        [0, 1]
    ])
    
    # Combined transformation: first scale, then shear, then rotate.
    M = R @ Sh @ S
    
    center = np.array(center)
    offset = center - M @ center + np.array([tx, ty])
    
    affine_matrix = np.hstack([M, offset.reshape(2,1)])
    return affine_matrix

def apply_affine(image, query_points, M):
    """
    Apply the affine transformation matrix M to an image and its query points.
    
    image: Input image as a NumPy array.
    query_points: Array of shape (N, 2) with (x,y) coordinates.
    M: 2x3 affine transformation matrix.
    
    Returns:
      transformed_image: Transformed image.
      transformed_points: Transformed query points (N, 2).
    """
    transformed_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((query_points.shape[0], 1))
    points_hom = np.hstack([query_points, ones])
    transformed_points = (M @ points_hom.T).T
    return transformed_image, transformed_points

def generate_interpolated_video(image, query_points, num_frames=5,
                                rotation_range=(-30,30), 
                                scale_range=(0.5,1.5),
                                shear_range=(-20,20), 
                                translation_range=(-50,50),
                                anisotropic=False,
                                scale_range_y=None):
    """
    Generate a synthetic video by applying a random affine transformation
    to the input image and its query points. The transformation is linearly
    interpolated over num_frames.
    
    Parameters:
      image: Input image (NumPy array, e.g. first frame).
      query_points: Array of shape (N, 2) with (x,y) coordinates.
      num_frames: Number of frames to generate (including start and end).
      anisotropic: If True, use independent scaling for x and y.
      scale_range_y: Tuple for y-scale if anisotropic is True.
      
    Returns:
      video_frames: List of transformed images.
      query_points_list: List of transformed query points (each shape: (N,2)).
    """
    angle, scale_x, scale_y, shear, tx, ty = random_affine_params(rotation_range, scale_range, shear_range, translation_range, anisotropic, scale_range_y)
    center = (image.shape[1] / 2, image.shape[0] / 2)
    target_M = get_affine_matrix(angle, scale_x, scale_y, shear, tx, ty, center)
    
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
    """
    Save a list of frames as a video.
    """
    import imageio
    imageio.mimwrite(output_path, frames, fps=fps)
    print("Video saved to", output_path)

def save_video_with_overlay(frames, query_points_list, output_path, fps=5,
                            point_color=(255,0,0), point_radius=3):
    """
    Draws query points on each frame and saves the video.
    
    Parameters:
      frames: List of frames (NumPy arrays).
      query_points_list: List of query point arrays (each shape: (N,2)).
      output_path: Path to save the video.
      fps: Frames per second.
      point_color: BGR color tuple for drawing points.
      point_radius: Radius of drawn points.
    """
    overlay_frames = []
    for frame, points in zip(frames, query_points_list):
        frame_overlay = frame.copy()
        for (x, y) in points:
            cv2.circle(frame_overlay, (int(x), int(y)), point_radius, point_color, -1)
        overlay_frames.append(frame_overlay)
    save_video(overlay_frames, output_path, fps=fps)
