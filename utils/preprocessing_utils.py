from PIL import Image
import open3d as o3d
import numpy as np
import trimesh
from trimesh.sample import sample_surface
from trimesh.resolvers import FilePathResolver
from .file_utils import load_pickle
import os

def retrieve_image_data_for_visualization(data_files, idx):
    rgb_files, depth_files, label_files, meta_files = data_files
    rgb = np.array(Image.open(rgb_files[idx])) / 255
    depth = np.array(Image.open(depth_files[idx])) / 1000
    label = np.array(Image.open(label_files[idx]))
    return rgb, depth, label

#%% md
## Lift depth to point cloud
#%%
def get_pc_from_image_files(rgb_file, depth_file, meta_file):
    meta = load_pickle(meta_file)
    intrinsic = meta['intrinsic']
    extrinsic = meta['extrinsic']
    rgb_image = o3d.io.read_image(rgb_file)
    depth_image = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_image,
                                                                    depth=depth_image,
                                                                    convert_rgb_to_intensity=False)
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = intrinsic
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                         intrinsic=cam,
                                                         extrinsic=extrinsic,
                                                         project_valid_depth_only=False)
    return pcd


'''
def lift_depth_to_pc(depth, rgb, meta):
    intrinsic = meta['intrinsic']
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    points = o3d.utility.Vector3dVector(points_viewer.reshape([-1, 3]))
    colors = o3d.utility.Vector3dVector(rgb.reshape([-1, 3]))
    pcd = o3d.geometry.PointCloud()
    pcd.points = points
    pcd.colors = colors
    return pcd
'''

def sample_pc_from_mesh_file(mesh_file, texture_file, npoints=1024):
    #print(mesh_file)
    #resolver = FilePathResolver(texture_file) if os.path.exists(texture_file) else None
    #scene = trimesh.load(mesh_file, resolver=resolver)
    scene = trimesh.load(mesh_file)
    mesh = scene.geometry[list(scene.geometry.keys())[0]]
    faces = mesh.faces
    vertex_colors = mesh.visual.to_color().vertex_colors
    #print(vertex_colors.shape)
    #print(vertex_colors[:5])
    #print(vertex_colors[-5:])
    #exit()
    #if not os.path.exists(texture_file):
    if len(vertex_colors.shape)==1:
        vertex_colors = np.tile(vertex_colors, (mesh.vertices.shape[0],1))
    face_colors = trimesh.visual.color.vertex_to_face_color(vertex_colors, faces)
    #face_colors = mesh.visual.to_color().face_colors
    pc, face_idx = sample_surface(mesh, npoints)
    pcolors= np.array(face_colors[face_idx], dtype=np.float32) / 255
    points = o3d.utility.Vector3dVector(pc.reshape([-1, 3]))
    colors = o3d.utility.Vector3dVector(pcolors[:,:3].reshape([-1, 3]))
    pcd = o3d.geometry.PointCloud()
    pcd.points = points
    pcd.colors = colors
    return pcd
