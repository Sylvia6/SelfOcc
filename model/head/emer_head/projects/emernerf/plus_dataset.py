from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from projects.emernerf.plus_dataparser import PlusDataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


#---------------- Cityscapes semantic segmentation
cityscapes_classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle"
]
cityscapes_classes_ind_map = {cn: i for i, cn in enumerate(cityscapes_classes)}

cityscapes_dynamic_classes = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

cityscapes_road_classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign"
]

cityscapes_human_classes = [
    "person", "rider"
]


class PlusDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask", 'dynamic_mask', 'sky_mask']
    cameras: Cameras

    def __init__(self, dataparser_outputs: PlusDataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx, "image": image}
        # if self._dataparser_outputs.norm_timestamps is not None:
        #     data['norm_timestamp'] = self._dataparser_outputs.norm_timestamps[image_idx]
        # handle mask
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filename = self._dataparser_outputs.mask_filenames[image_idx].as_posix()
            if mask_filename is not None:
                if mask_filename is not None:
                    if mask_filename.endswith(".npy"):
                        raw = np.load(mask_filename)
                        # ret = np.zeros_like(raw).astype(np.bool8)
                        # for cls in cityscapes_dynamic_classes:
                        #     ind = cityscapes_classes_ind_map[cls]
                        #     ret[raw==ind] = True                    
                    elif mask_filename.endswith(".png"):
                        raw = np.array(Image.open(mask_filename).convert("L"))
                    else:
                        raise TypeError("Error type of mask_file")
                    raw = raw.squeeze()
                    dynamic_mask = np.zeros_like(raw).astype(np.uint8)
                    for cls in cityscapes_dynamic_classes:
                        ind = cityscapes_classes_ind_map[cls]
                        dynamic_mask[raw==ind] = 1  

                    sky_mask = np.zeros_like(raw).astype(np.uint8)
                    sky_mask[raw==cityscapes_classes_ind_map["sky"]] = 1
                    dynamic_mask = torch.from_numpy(np.array(dynamic_mask)).unsqueeze(-1).bool().long()
                    sky_mask = torch.from_numpy(np.array(dynamic_mask)).unsqueeze(-1).bool().long()
                    data['dynamic_mask'] = dynamic_mask
                    data['sky_mask'] = sky_mask
        return data

    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames

    def get_training_unique_timestamps(self):
        timestamps = self._dataparser_outputs.norm_timestamps.unique()
        time_diff = 1 / len(timestamps.unique())
        return timestamps, time_diff
