import random
import numpy as np, torch
from torch.utils import data
from dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage, \
    RandomFlip
import torch.nn.functional as F
from copy import deepcopy
from mmengine import MMLogger
logger = MMLogger.get_instance('selfocc')
from dataset.dataset_wrapper import forward_aug, img_norm_cfg
from . import OPENOCC_DATAWRAPPER


@OPENOCC_DATAWRAPPER.register_module()
class simpleocc_dataset_plusai_temporal(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
            scale_rate=1,
            input_length=3,
            prev_length=5,
            post_length=5,
            photometric_aug=dict(
                use_swap_channel=False,
            ),
            use_temporal_aug=False,
            temporal_aug_list=[],
            img_norm_cfg=img_norm_cfg,
            supervision_img_size=None,
            supervision_scale_rate=None,
            use_flip=False,
            ref_focal_len=None,
            pad_img_size=None,
            random_scale=None,
            pad_scale_rate=None,
        ):
        'Initialization'
        self.dataset = in_dataset
        self.prev_length = prev_length
        self.post_length = post_length
        self.scale_rate = scale_rate
        self.use_temporal_aug = use_temporal_aug
        if use_temporal_aug:
            assert len(temporal_aug_list) > 0
        self.temporal_aug_list = temporal_aug_list

        photometric = PhotoMetricDistortionMultiViewImage(**photometric_aug)
        logger.info('using photometric augmentation: '+ str(photometric_aug))

        train_transforms = [
            photometric,
            NormalizeMultiviewImage(**img_norm_cfg),
            PadMultiViewImage(size_divisor=32, size=pad_img_size)
        ]
        val_transforms = [
            NormalizeMultiviewImage(**img_norm_cfg),
            PadMultiViewImage(size_divisor=32, size=pad_img_size)
        ]
        if scale_rate != 1 or ref_focal_len is not None or random_scale is not None or pad_scale_rate is not None:
            train_transforms.insert(2, RandomScaleImageMultiViewImage([scale_rate], ref_focal_len, random_scale, pad_scale_rate))
            val_transforms.insert(1, RandomScaleImageMultiViewImage([scale_rate], ref_focal_len, pad_scale_rate=pad_scale_rate))
        if use_flip:
            train_transforms.append(RandomFlip(0.5))
        
        self.phase = phase
        if phase == 'train':
            self.transforms = train_transforms
        else:
            self.transforms = val_transforms
        if use_temporal_aug:
            self.temporal_transforms = [
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=4)]
            if supervision_scale_rate != 1:
                self.temporal_transforms.insert(1, RandomScaleImageMultiViewImage([supervision_scale_rate]))
        self.supervision_img_size = supervision_img_size


    def __len__(self):
        return len(self.dataset) - self.input_length + 1


    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)


    def __getitem__(self, index):
        """
        Returns:
            imgs: Tensor[T, N, C, H, W]
            gt_imgs: Tensor[N_gt, C, H, W]
            metas: Dict(
                input:
                    t_frames: int
                    n_imgs: int
                    timestamps: Tensor(T)
                    cur_timestamp: float
                    relative_timestamps: Tensor(T)  # relative to cur_timestamp
                    cam2imu: Tensor(T, 4, 4)
                    imu2world: Tensor(T, 4, 4)
                    cam2pixel: Tensor(T, N, 3, 4)
                gt:
                    n_imgs: int
                    timestamps: Tensor(N_gt)
                    relative_timestamps: Tensor(N_gt) # relative to cur_timestamp
                    cam2imu: Tensor(N_gt, 4, 4)
                    imu2world: Tensor(N_gt, 4, 4)
                    cam2pixel: Tensor(N_gt, 3, 3)
                    world2pixel: Tensor(N_gt, 3, 4)

            )
        """
        # ======= input =================
        def concat_array(array, transforms):
            meta = dict()
            imgs = [x[0] for x in array]
            imgs, _ = forward_aug(imgs, {}, transforms)
            imgs = self.to_tensor(imgs) # T, N, 3, H, W
            input_metas = [x[1] for x in array]
            timestamps = [input_meta["timestamp"] for input_meta in input_metas]
            meta["timestamps"] = torch.Tensor(timestamps, dtype=torch.float32)
            meta["cur_timestamp"] = meta["timestamps"][-1]
            meta["relative_timestamps"] = meta["timestamps"] - meta["cur_timestamp"]
            meta["cam2imu"] = torch.stack([torch.from_numpy(input_meta["cam2imu"]) for input_meta in input_metas])
            meta["imu2world"] = torch.stack([torch.from_numpy(input_meta["imu2world"]) for input_meta in input_metas])
            meta["cam2pixel"] = torch.stack([torch.from_numpy(input_meta["cam2pixel"]) for input_meta in input_metas])
            return imgs, meta

        meta = dict()
        inputs = []
        for i in range(index - self.input_length + 1, index + 1):
            inputs.append(self.dataset[i])
        imgs, input_meta = concat_array(inputs, self.transforms)
        input_meta["t_frames"] = self.input_length
        input_meta["n_imgs"] = inputs[0][1]["n_imgs"]
        meta["input"] = input_meta

        # ======= gt =================
        gts = []
        # length = prev_length + post_length + 1
        for i in range(index - self.prev_length , index + self.post_length + 1):
            # 如果超出dataset的长度，就取最后一个 
            idx = i if i < len(self.dataset) else len(self.dataset) - 1
            # 如果小于0，就取第一个
            idx = i if i >= 0 else 0
            gts.append(self.dataset[idx])
        gt_imgs, gt_meta = concat_array(gts, self.temporal_transforms, "gt")
        # random pick N_gt
        T, N = gt_imgs.shape[:2]
        assert T*N >= self.N_gt
        idxes = random.sample(range(T*N), self.N_gt)
        gt_imgs = gt_imgs.view(T*N, *gt_imgs.shape[2:])[idxes]
        gt_meta["timestamps"] = gt_meta["timestamps"].unsqueeze(1).repeat(1, N).view(T*N)[idxes]
        gt_meta["relative_timestamps"] = gt_meta["timestamps"] - input_meta["cur_timestamp"]
        gt_meta["relative_timestamps"] = gt_meta["relative_timestamps"].unsqueeze(1).repeat(1, N).view(T*N)[idxes]
        gt_meta["cam2imu"] = gt_meta["cam2imu"].unsqueeze(1).repeat(1, N).view(T*N, *gt_meta["cam2imu"].shape[2:])[idxes]
        gt_meta["cam2world"] = gt_meta["cam2world"].unsqueeze(1).repeat(1, N).view(T*N, *gt_meta["cam2world"].shape[2:])[idxes]
        gt_meta["cam2pixel"] = gt_meta["cam2pixel"].view(T*N)[idxes]
        meta["gt"] = gt_meta
        return imgs, gt_imgs, meta