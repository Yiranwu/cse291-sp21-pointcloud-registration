import torch
import numpy as np
from pathlib import Path
import numpy as np
import cv2
from transforms3d.quaternions import quat2mat
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from .preprocessing_utils import retrieve_image_data_for_visualization
from .file_utils import load_pickle
from PIL import Image
import open3d as o3d


def visualize_pc(pcd):
    o3d.visualization.draw_geometries([pcd])

def get_palette():
    NUM_OBJECTS = 79
    cmap = get_cmap('rainbow', NUM_OBJECTS)
    COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(NUM_OBJECTS + 3)])
    COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
    COLOR_PALETTE[-3] = [119, 135, 150]
    COLOR_PALETTE[-2] = [176, 194, 216]
    COLOR_PALETTE[-1] = [255, 255, 225]
    return COLOR_PALETTE

def visualize_data_image(data_files):
    rgb, depth, label = retrieve_image_data_for_visualization(data_files, 0)
    COLOR_PALETTE = get_palette()
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.subplot(1, 3, 2)
    plt.imshow(depth)
    plt.subplot(1, 3, 3)
    plt.imshow(COLOR_PALETTE[label])  # draw colorful segmentation

def draw_bbox(data_files):
    ## Draw bounding boxes of poses on 2D image
    rgb_files, depth_files, label_files, meta_files = data_files
    rgb = load_pickle(rgb_files[0])
    meta = load_pickle(meta_files[0])
    meta.keys()
    #meta['object_names'], meta['object_ids']
    poses_world = np.array([meta['poses_world'][idx] for idx in meta['object_ids']])
    box_sizes = np.array([meta['extents'][idx] * meta['scales'][idx] for idx in meta['object_ids']])
    boxed_image = np.array(rgb)
    for i in range(len(poses_world)):
        draw_projected_box3d(
            boxed_image, poses_world[i][:3, 3], box_sizes[i], poses_world[i][:3, :3], meta['extrinsic'],
            meta['intrinsic'],
            thickness=2)

    img = Image.fromarray((boxed_image * 255).astype(np.uint8))
    img.show()

VERTEX_COLORS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]


def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)
        (y)
        2 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 0 -------- 1 (x)
      |/         |/
      3 -------- 6
      (z)
    """
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def draw_projected_box3d(
    image, center, size, rotation, extrinsic, intrinsic, color=(0, 1, 0), thickness=1
):
    """Draw a projected 3D bounding box on the image.

    Args:
        image (np.ndarray): [H, W, 3] array.
        center: [3]
        size: [3]
        rotation (np.ndarray): [3, 3]
        extrinsic (np.ndarray): [4, 4]
        intrinsic (np.ndarray): [3, 3]
        color: [3]
        thickness (int): thickness of lines
    Returns:
        np.ndarray: updated image.
    """
    corners = get_corners()  # [8, 3]
    edges = get_edges(corners)  # [12, 2]
    corners = corners * size
    corners_world = corners @ rotation.T + center
    corners_camera = corners_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    corners_image = corners_camera @ intrinsic.T
    uv = corners_image[:, 0:2] / corners_image[:, 2:]
    uv = uv.astype(int)

    for (i, j) in edges:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            tuple(color),
            thickness,
            cv2.LINE_AA,
        )

    for i, (u, v) in enumerate(uv):
        cv2.circle(image, (u, v), radius=1, color=VERTEX_COLORS[i], thickness=1)
    return image
