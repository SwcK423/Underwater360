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
import math

import kornia
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.image_utils import get_img_grad_weight


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, image, gt_alpha_mask,
                 uid, image_name, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.nearest_id = []
        self.nearest_names = []
        self.colmap_id = colmap_id
        self.R = torch.tensor(R).float().cuda()
        self.T = torch.tensor(T).float().cuda()
        self.image_name = image_name

        # W2C = np.eye(4)
        # W2C[:3, :3] = np.transpose(R)
        # W2C[:3, 3] = T
        # C2W = np.linalg.inv(W2C)
        # self.cam_position = torch.tensor(C2W[:3, 3]).float().cuda()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0) # move to device at dataloader to reduce VRAM requirement
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device) # do we need this?
            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width))

        self.focal_x = self.image_width / (2 * math.pi)
        self.focal_y = self.image_height / math.pi

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = torch.eye(4).cuda()  # no use
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.grid = kornia.utils.create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device='cuda')[0]

    def get_world_directions(self, train=False):
        u, v = self.grid.unbind(-1)
        if train:
            directions = torch.stack([(u - self.cx + torch.rand_like(u)) / self.fx,
                                      (v - self.cy + torch.rand_like(v)) / self.fy,
                                      torch.ones_like(u)], dim=0)
        else:
            directions = torch.stack([(u - self.cx + 0.5) / self.fx,
                                      (v - self.cy + 0.5) / self.fy,
                                      torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions

    def get_world_directions_panorama(self):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        theta = (theta / (self.image_height - 1)) * torch.pi
        phi = (2 * phi / (self.image_width - 1)) * torch.pi - torch.pi

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        directions = torch.stack([dx, dy, dz], dim=0)
        directions = F.normalize(directions, p=2, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions

    def get_local_directions_panorama(self, train=False):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        theta = (theta / self.image_height) * torch.pi
        phi = (2 * phi / self.image_width - 1) * torch.pi

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        directions = torch.stack([dx, dy, dz], dim=0)
        directions = F.normalize(directions, dim=0)
        return directions

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

