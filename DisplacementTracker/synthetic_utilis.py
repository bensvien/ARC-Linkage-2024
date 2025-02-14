# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:16:44 2025

@author: Dr. Benjamin Vien
"""

# synthetic_utils.py
import cv2
import numpy as np
import torch
import imageio.v3 as iio
import scipy.io as sio
import matplotlib.pyplot as plt

def random_affine_params(rotation_range=(-30, 30), 
                           scale_range=(1.2, 1.5), 
                           shear_x_range=(-10, 10), 
                           shear_y_range=(-10, 10),
                           translation_range=(-20, 20),
                           anisotropic=False,
                           scale_range_y=(0.8, 1.0)):
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
    transformed_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((query_points.shape[0], 1))
    points_hom = np.hstack([query_points, ones])
    transformed_points = (M @ points_hom.T).T
    return transformed_image, transformed_points

def generate_interpolated_video(image, query_points, num_frames=5,
                                rotation_range=(-30,30), 
                                scale_range=(1.2,1.5),
                                shear_x_range=(-10,10), 
                                shear_y_range=(-10,10),
                                translation_range=(-20,20),
                                anisotropic=False,
                                scale_range_y=(0.8,1.0)):
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
