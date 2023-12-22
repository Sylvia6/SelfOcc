import pickle, cv2
from tqdm import tqdm
import json
import numpy as np
import torch
import os
from PIL import Image
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from projects.emernerf.emer_nerf_camera import EmerCamera
from nerfstudio.data.scene_box import SceneBox
import math
import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor


from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
    get_train_eval_split_all,
)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def cal_shadow_mask(ret, ratio=[1.2, 1.0]):
    mask = ret.astype(np.uint8) * 255   # h,w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_shift = x - (w * (ratio[0]-1)) / 2
        x_min, y_min = int(max(0, x_shift)), int(max(0, y))
        x_max, y_max = int(min(mask.shape[1], x_shift + ratio[1] * w)), int(min(mask.shape[0], y + ratio[0] * h))
        ret[y_min:y_max, x_min:x_max] = True
    return ret


def cam_pose_to_nerf(cam_pose, gl=True):  
    """
        plus_data的camera为opencv坐标系
        
        emernerf的数据坐标输入为opencv坐标,nerfstudio的数据坐标输入为opengl坐标
        
        nerf_world与eqdc坐标系一样
        
        < opencv / colmap convention >                --->>>     < opengl / NeRF convention >                    --->>>   < world convention >
        facing [+z] direction, x right, y downwards   --->>>    facing [-z] direction, x right, y upwards        --->>>  facing [+x] direction, z upwards, y left
                    z                                              y ↑                                                      z ↑    x
                   ↗                                                 |                                                        |   ↗ 
                  /                                                  |                                                        |  /
                 /                                                   |                                                        | /
                o------> x                                           o------> x                                    y ←--------o
                |                                                   /
                |                                                  /
                |                                               z ↙
                ↓ 
                y
    """
    if gl:
        opencv2opengl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0],[0,0,0,1]]).astype(float)
        opengl2opencv = np.linalg.inv(opencv2opengl)
        opengl2world = np.array([[0,0,-1,0], [-1,0,0,0], [0,1,0,0],[0,0,0,1]]).astype(float)
        gl_pose = opencv2opengl @ cam_pose @ opengl2opencv  
        world_pose = opengl2world @ gl_pose
    else:
        opencv2world = np.array([[0,0,1,0], [-1,0,0,0], [0,-1,0,0],[0,0,0,1]]).astype(float)
        world_pose = opencv2world @ cam_pose
    return world_pose


total_camera_dict = {cam:i for i, cam in enumerate(["front_left", "front_right"])}


@dataclass
class PlusDataparserOutputs(DataparserOutputs):
    norm_timestamps: Optional[torch.Tensor] = None
    # sky_mask_filenames: Optional[List[Path]] = None
    # dynamic_mask_filenames: Optional[List[Path]] = None


@dataclass
class PlusDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: PlusDataParser)
    """target class to instantiate"""
    data_path: str = ""
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval. 
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    num_cams: int = 2
    start_timestep: int = 0
    end_timestep: int = -1    
    pose_type: str = 'vio'



def auto_orient_and_center_poses(
    poses: Float[Tensor, "*num_poses 4 4"],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> Tuple[Float[Tensor, "*num_poses 3 4"], Float[Tensor, "3 4"]]:

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = camera_utils.focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            # If cameras are not all parallel (e.g. not in an LLFF configuration),
            # we can find the 3D direction that most projects vertically in all
            # cameras by minimizing ||Xu|| s.t. ||u||=1. This total least squares
            # problem is solved by SVD.
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            # Singular values are S_i=||Xv_i|| for each right singular vector v_i.
            # ||S|| = sqrt(n) because lines of X are all unit vectors and the v_i
            # are an orthonormal basis.
            # ||Xv_i|| = sqrt(sum(dot(x_axis_j,v_i)^2)), thus S_i/sqrt(n) is the
            # RMS of cosines between x axes and v_i. If the second smallest singular
            # value corresponds to an angle error less than 10° (cos(80°)=0.17),
            # this is probably a degenerate camera configuration (typical values
            # are around 5° average error for the true vertical). In this case,
            # rather than taking the vector corresponding to the smallest singular
            # value, we project the "up" vector on the plane spanned by the two
            # best singular vectors. We could also just fallback to the "up"
            # solution.
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                # regular non-degenerate configuration
                up_vertical = Vh[2, :]
                # It may be pointing up or down. Use "up" to disambiguate the sign.
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                # Degenerate configuration: project "up" on the plane spanned by
                # the last two right singular vectors (which are orthogonal to the
                # first). v_0 is a unit vector, no need to divide by its norm when
                # projecting.
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                # re-normalize
                up = up / torch.linalg.norm(up)

        rotation = camera_utils.rotation_matrix(up.float(), torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None].float()], dim=-1)
        transform = transform.type_as(poses)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform


@dataclass
class PlusDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: PlusDataParserConfig
    downscale_factor: Optional[int] = None

    def __post_init__(self):

        if self.config.end_timestep == -1:
            files = sorted(os.listdir(os.path.join(self.config.data_path, "images", "front_left")))
            end_timestep = len(files)    # 有时候vio-pose会丢帧
        else:
            end_timestep = self.config.end_timestep
        # to make sure the last timestep is included
        # end_timestep = end_timestep + 1
        self.config.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_calibrations()

    def create_all_filelist(self):
        """
        Create file lists for all data files.
        e.g., img files, feature files, etc.
        """
        if self.config.num_cams == 1:
            self.camera_list = ["front_left",]
        elif self.config.num_cams == 2:
            self.camera_list = ["front_left", "front_right"]
        else:
            raise NotImplementedError(
                f"num_cams: {self.config.num_cams} not supported for plus dataset"
            )

        # ---- define filepaths ---- #
        img_filepaths, feat_filepaths = [], []
        mask_filepaths = []
        selected_steps = []    # 可用帧的数据在self.start_timestep到self.end_timestep中的indice

        # Note: we assume all the files in plus dataset are synchronized
        for t in range(self.config.start_timestep, self.config.end_timestep):
            for cam_i in self.camera_list:
                img_filepath = os.path.join(self.config.data_path, "images", cam_i, f"{t:06d}.png")
                if os.path.exists(img_filepath):
                    img_filepaths.append(img_filepath)
                    selected_steps.append(t)
                mask_filepath = os.path.join(self.config.data_path, "images", f"{cam_i}_mask", f"{t:06d}.npy")
                if not os.path.exists(mask_filepath):
                    mask_filepath = os.path.join(self.config.data_path, "box_masks", cam_i, f"{t:06d}.png")
                if not os.path.exists(mask_filepath):
                    mask_filepaths.append(None)
                else:
                    mask_filepaths.append(mask_filepath)

        self.image_filenames = img_filepaths
        self.mask_filenames = mask_filepaths
        self.selected_steps = np.array(selected_steps)

    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # to store per-camera intrinsics and extrinsics

        # compute per-image poses and intrinsics
        data_file = os.path.join(self.config.data_path, "transforms.json")
        assert os.path.exists(data_file)
        with open(data_file) as f:
            data = json.load(f)

        frames = data["frames"]
        cam_ids = np.array([total_camera_dict[frame["cam_name"]] for frame in frames])
        intrs = np.array([frame["intr"] for frame in frames])

        if self.config.pose_type == 'odom':
            c2ws = np.array([frame["transform_matrix"] for frame in frames]) 
        elif self.config.pose_type == 'vio':
            # c2ws = np.array([cam_pose_to_nerf(frame["transform_matrix_vio"], gl=True) for frame in frames])
            c2ws = np.array([frame["transform_matrix_vio"] for frame in frames]) 
        else:
            raise ValueError("The pose_type is a ValueError str.")

        indices = self.selected_steps * self.config.num_cams
        for i in range(self.config.num_cams):
            indices[i::self.config.num_cams] += i

        self.intrinsics = torch.from_numpy(intrs).float()[indices]
        self.cam_to_worlds = torch.from_numpy(c2ws)[indices]    # 避免损失eqdc的精度   .float()
        # self.ego_to_worlds = torch.from_numpy(poses_imu_w_tracking).float()
        self.cam_ids = torch.from_numpy(cam_ids).long()[indices]

        # the underscore here is important.
        self.timestamps = torch.from_numpy(self.selected_steps-self.config.start_timestep).float()
        self.timesteps = torch.from_numpy(self.selected_steps-self.config.start_timestep).long()
        # print(self.timestamps.shape)
        self.norm_timestamps = self.timestamps / self.timestamps.max()

    def cal_aabb(self):
        if self.config.num_cams == 1:
            # if there is only one camera, it's front camera
            front_cameras_positions = self.cam_to_worlds[:, :3, 3]
        elif self.config.num_cams == 2:
            # if there are two cameras, they are ordered as front_left, front_right
            front_cameras_positions = self.cam_to_worlds[::2, :3, 3]

        aabb_min = front_cameras_positions.min(dim=0)[0]
        aabb_max = front_cameras_positions.max(dim=0)[0]

        # extend aabb by 40 meters along forward direction and 40 meters along the left/right direction
        # aabb direction: x, y, z: front, left, up
        aabb_max[0] += 40
        aabb_max[1] += 40
        # when the car is driving uphills
        aabb_max[2] = max(aabb_max[2] + 20, 20)

        # for waymo, there will be a lot of waste of space because we don't have images in the back,
        # it's more reasonable to extend the aabb only by a small amount, e.g., 5 meters
        # we use 40 meters here for a more general case
        aabb_min[0] -= 40
        aabb_min[1] -= 40
        # when a car is driving downhills
        aabb_min[2] = min(aabb_min[2] - 5, -5)
        aabb = torch.tensor([*aabb_min, *aabb_max])
        return aabb

    def center_pose(self, poses):
        aabb = self.cal_aabb()
        
        pass

    def _generate_dataparser_outputs(self, split="train"):
        i_train, i_eval = get_train_eval_split_fraction(self.image_filenames[::self.config.num_cams], self.config.train_split_fraction)

        def gen_idx(old_idxs, cam_num):
            new_idxs = []
            for idx in old_idxs:
                new_idxs.extend([idx * cam_num + i for i in range(cam_num)])
            return new_idxs
        
        i_train = gen_idx(i_train, self.config.num_cams)
        i_eval = gen_idx(i_eval, self.config.num_cams)

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = self.cam_to_worlds[indices]
        # orientation_method = self.config.orientation_method
        # poses, transform_matrix = auto_orient_and_center_poses(
        #     poses,
        #     method=orientation_method,
        #     center_method=self.config.center_method,
        # )
        # poses = poses.float()
        #         # Scale poses
        # scale_factor = 1.0
        # if self.config.auto_scale_poses:
        #     scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        # scale_factor *= self.config.scale_factor

        # poses[:, :3, 3] *= scale_factor
        
        # aabb_scale = self.config.scene_scale
        # scene_box = SceneBox(
        #     aabb=torch.tensor(
        #         [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
        #     )
        # )

        poses = poses.float()
        aabb = self.cal_aabb()
        # scale_factor = 1.0 / float(torch.max(torch.abs(aabb)))
        scale_factor = 1.0
        poses[:, :3, 3] *= scale_factor
        aabb = aabb * scale_factor
        scene_box = SceneBox(aabb=aabb.view(2, 3).float())
        transform_matrix = torch.eye(4)

        fx = self.intrinsics[indices][:, 0, 0]
        fy = self.intrinsics[indices][:, 1, 1]
        cx = self.intrinsics[indices][:, 0, 2]
        cy = self.intrinsics[indices][:, 1, 2]
        norm_timestamps = self.norm_timestamps[indices]
        # TODO: ??? should not be fixed
        height = torch.tensor([540 for _ in range(len(fx))])
        width = torch.tensor([960 for _ in range(len(fx))])

        distort = camera_utils.get_distortion_params( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        distortion_params = distort.repeat(len(fx), 1)
        
        camera_type = CameraType.PERSPECTIVE

        cameras = EmerCamera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            times=norm_timestamps
        )

        # image_filenames = self.image_filenames[indices]
        image_filenames = [Path(self.image_filenames[i]) for i in indices]
        # mask_filenames = self.mask_filenames[indices]
        mask_filenames = [Path(self.mask_filenames[i]) for i in indices]
        dataparser_outputs = PlusDataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
            norm_timestamps=norm_timestamps
        )
        return dataparser_outputs
