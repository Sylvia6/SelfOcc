from typing import Optional, Union, Tuple
from jaxtyping import Float, Int
from torch import Tensor
from nerfstudio.cameras.cameras import Cameras, dataclass
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox, SceneBox
import torch


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
    directions = (camera_dirs[..., None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    # TODO: not sure if we still need direction_norm
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    # normalize the ray directions
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm


@dataclass
class EmerCamera(Cameras):
    def __init__(self, *argc, **kargv):
        super().__init__(*argc, **kargv)

    def generate_rays(
        self,
        camera_indices: Union[Int[Tensor, "*num_rays num_cameras_batch_dims"], int],
        coords: Optional[Float[Tensor, "*num_rays 2"]] = None,
        camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
        distortion_params_delta: Optional[Float[Tensor, "*num_rays 6"]] = None,
        keep_shape: Optional[bool] = None,
        disable_distortion: bool = False,
        aabb_box: Optional[SceneBox] = None,
        obb_box: Optional[OrientedBox] = None,
    ) -> RayBundle:

        if not self.shape:
            cameras = self.reshape((1,))
            assert torch.all(
                torch.tensor(camera_indices == 0) if isinstance(camera_indices, int) else camera_indices == 0
            ), "Can only index into single camera with no batch dimensions if index is zero"
        else:
            cameras = self

        if isinstance(camera_indices, int):
            assert (
                len(cameras.shape) == 1
            ), "camera_indices must be a tensor if cameras are batched with more than 1 batch dimension"
            camera_indices = torch.tensor([camera_indices])

        if cameras.is_jagged and coords is None and (keep_shape is None or keep_shape is False):
            _coords = [cameras.get_image_coords(index=tuple(index)).reshape(-1, 2) for index in camera_indices]
            camera_indices = torch.cat(
                [index.unsqueeze(0).repeat(coords.shape[0], 1) for index, coords in zip(camera_indices, _coords)],
            )
            coords = torch.cat(_coords, dim=0)

        if coords is None:
            index_dim = camera_indices.shape[-1]
            index = camera_indices.reshape(-1, index_dim)[0]
            coords = cameras.get_image_coords(index=tuple(index))  # (h, w, 2)
            coords = coords.reshape(coords.shape[:2] + (1,) * len(camera_indices.shape[:-1]) + (2,))  # (h, w, 1..., 2)
            coords = coords.expand(coords.shape[:2] + camera_indices.shape[:-1] + (2,))  # (h, w, num_rays, 2)
            camera_opt_to_camera = (  # (h, w, num_rays, 3, 4) or None
                camera_opt_to_camera.broadcast_to(coords.shape[:-1] + (3, 4))
                if camera_opt_to_camera is not None
                else None
            )
            distortion_params_delta = (  # (h, w, num_rays, 6) or None
                distortion_params_delta.broadcast_to(coords.shape[:-1] + (6,))
                if distortion_params_delta is not None
                else None
            )

        y, x = coords[..., 0], coords[..., 1]
        if len(camera_indices.shape) == 2:
            camera_indices = camera_indices.squeeze(-1)
        c2w = cameras.camera_to_worlds[camera_indices]
        intrinsics = cameras.get_intrinsics_matrices()[camera_indices].to(c2w.device)
        
        origins, viewdirs, direction_norm = get_rays(x.to(c2w.device), y.to(c2w.device), c2w, intrinsics)

        return RayBundle(
            origins=origins,
            directions=viewdirs,
            pixel_area=torch.ones_like(origins[..., :1]),
            camera_indices=camera_indices.unsqueeze(-1),
            times=None if cameras.times is None else cameras.times[camera_indices]
        )
