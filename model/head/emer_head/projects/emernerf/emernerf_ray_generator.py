# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ray generator.
"""
from jaxtyping import Int
from torch import Tensor, nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
import torch
from typing import Tuple


def get_rays(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        x: the horizontal coordinates of the pixels, shape: (num_rays,)
        y: the vertical coordinates of the pixels, shape: (num_rays,)
        c2w: the camera-to-world matrices, shape: (num_cams, 4, 4)
        intrinsic: the camera intrinsic matrices, shape: (num_cams, 3, 3)
    Returns:
        origins: the ray origins, shape: (num_rays, 3)
        viewdirs: the ray directions, shape: (num_rays, 3)
        direction_norm: the norm of the ray directions, shape: (num_rays, 1)
    """
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]
    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [num_rays, 3]

    # rotate the camera rays w.r.t. the camera pose
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    # TODO: not sure if we still need direction_norm
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    # normalize the ray directions
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm


class EmerRayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        cameras: Camera objects containing camera info.
    """

    image_coords: Tensor

    def __init__(self, cameras: Cameras) -> None:
        super().__init__()
        self.cameras = cameras
        self.register_buffer("image_coords", cameras.get_image_coords(), persistent=False)

    def forward(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices

        c2w = self.cameras.camera_to_worlds[c]
        intrinsics = self.cameras.get_intrinsics_matrices()[c].to(c2w.device)
        origins, viewdirs, direction_norm = get_rays(x.to(c2w.device), y.to(c2w.device), c2w, intrinsics)

        return RayBundle(
            origins=origins,
            directions=viewdirs,
            pixel_area=torch.ones_like(origins[..., :1]),
            camera_indices=c,
            times=None if self.cameras.times is None else self.cameras.times[c]
        )
