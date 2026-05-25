#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

import cv2
import kornia
import torch
import torchvision
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement

from utils.image_utils import get_img_grad_weight
from utils.point_utils import depth_to_normal
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

import open3d as o3d
import torch.nn.functional as F
import xml.etree.ElementTree as ET

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path, world_system=None):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # positions = np.vstack([vertices['x'], -vertices['z'], vertices['y']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def get_panorama_rays(H=1000, W=2000):
    # get reference camera
    theta, phi = torch.meshgrid(torch.arange(H, device='cuda'),
                                torch.arange(W, device='cuda'), indexing="ij")

    theta = (theta / (H - 1)) * torch.pi
    phi = (2 * phi / (W - 1)) * torch.pi - torch.pi

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = torch.cos(theta)

    ray_dirs = torch.stack([dx, dy, dz], dim=-1)
    ray_dirs = F.normalize(ray_dirs, dim=-1)

    return ray_dirs


def readCamerasFromOpenMVG(path, extrinsicsfile, cam_dict, white_background):
    cam_infos = []

    mask_path = os.path.join(path, "masks")
    depth_path = os.path.join(path, "depths")
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)

    with open(os.path.join(path, extrinsicsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]
        # fovx = 1.59451063 # 0.8279103882874479
        fovx = 3.13768641

        frames = contents["extrinsics"]
        for idx, frame in enumerate(frames):
            cam_key = frame["key"]
            cam_name = os.path.join(path, 'images', cam_dict[cam_key])

            R = np.array(frame["value"]["rotation"]).T
            T = -np.array(frame["value"]["rotation"]) @ np.array(frame["value"]["center"])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, mask=None, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readOpenMVGInfo(path, white_background, eval):
    print("Reading Transforms from OpenMVG")

    my_views = os.path.join(path, "data_views.json")
    camfile_dict = {}
    with open(my_views) as views:
        json_views = json.load(views)
        camview_list = json_views["views"]
        for camview in camview_list:
            camfile_dict[camview["key"]] = camview["value"]["ptr_wrapper"]["data"]["filename"]

    cam_infos_unsorted = readCamerasFromOpenMVG(path, "data_extrinsics.json", camfile_dict, white_background)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    try:
        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')
        with open(train_file, 'r') as f:
            train_name_list = f.read().splitlines()
        with open(test_file, 'r') as f:
            test_name_list = f.read().splitlines()
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_name_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_name_list]

    except:
        raise AssertionError("Please Specify train test split")

    print(f"# of Train: {len(train_cam_infos)}, \t# of Test: {len(test_cam_infos)}")

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if os.path.exists(os.path.join(path, "pcd.ply")):
        print("Points without camera position (Green points) are initialized")
        ply_path = os.path.join(path, "pcd.ply")
    else:
        ply_path = os.path.join(path, "colorized.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError('No initial pcd file found!')
        # # Since this data set has no colmap data, we start with random points
        # num_pts = 100_000
        # print(f"Generating random point cloud ({num_pts})...")

        # # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        # shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    # c = readCubemapDepths(path, cam_infos, pcd)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readOmniUWInfo(path, white_background, eval):
    print("Reading Transforms from OmniUW")

    fovx = 3.13768641
    cam_infos = []

    my_views = os.path.join(path, "transforms.xml")
    tree = ET.parse(my_views)
    root = tree.getroot()

    def camera_system(x, y, z):
        return [x, y, -z]

    def world_system(x, y, z):
        return [x, y, -z]

    for child0 in root:
        for grandchild in child0:
            if grandchild.tag == 'cameras':
                for cam in grandchild:
                    id = int(cam.attrib['id'])
                    image_name = cam.attrib['label']
                    image_path = os.path.join(path, 'images', f'{image_name}.png')

                    trans = cam.find('transform').text.split(' ')
                    trans = np.array([float(i) for i in trans]).reshape(4, 4)

                    c2w = torch.from_numpy(trans)
                    # apply camera coordinate system conversion
                    if camera_system is not None:
                        c2w[:3, :3] = torch.cat(camera_system(*torch.split(c2w[:3, :3], split_size_or_sections=1, dim=1)), dim=1)
                    # apply world coordinate system conversion
                    if world_system is not None:
                        c2w[:3, :] = torch.cat(world_system(*torch.split(c2w[:3, :], split_size_or_sections=1, dim=0)), dim=0)

                    W2C = np.linalg.inv(c2w.numpy())

                    R = np.transpose(W2C[:3, :3])
                    T = W2C[:3, 3]

                    try:
                        image = Image.open(image_path)
                    except:
                        try:
                            image_path = os.path.join(path, 'images', f'{image_name}.jpg')
                            image = Image.open(image_path)
                        except:
                            image_path = os.path.join(path, 'images', f'{image_name}.jpeg')
                            image = Image.open(image_path)

                    im_data = np.array(image.convert("RGBA"))
                    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

                    norm_data = im_data / 255.0
                    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                    image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

                    cam_infos.append(CameraInfo(uid=id, R=R, T=T,
                                                image=image,
                                                image_path=image_path, image_name=image_name,
                                                width=image.size[0], height=image.size[1]))

    try:
        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')
        with open(train_file, 'r') as f:
            train_name_list = f.read().splitlines()
        with open(test_file, 'r') as f:
            test_name_list = f.read().splitlines()
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_name_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_name_list]

    except:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.uid % 2 == 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.uid % 2 == 1]

    print(f"# of Train: {len(train_cam_infos)}, \t# of Test: {len(test_cam_infos)}")

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if os.path.exists(os.path.join(path, "pcd.ply")):
        print("Points without camera position (Green points) are initialized")
        ply_path = os.path.join(path, "pcd.ply")
    else:
        ply_path = os.path.join(path, "colorized.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError('No initial pcd file found!')

    pcd = fetchPly(ply_path, world_system)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "OpenMVG" : readOpenMVGInfo,
    "OmniUW" : readOmniUWInfo,
}