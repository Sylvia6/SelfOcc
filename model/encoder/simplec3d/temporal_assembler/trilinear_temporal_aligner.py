import torch

from mmengine.registry import MODELS
from mmengine.model import BaseModule

from .... import utils

@MODELS.register_module()
class TrilinearTemporalAligner(BaseModule):
    def __init__(self, T, nZ, nY, nX, bounds, output_mapper):
        super(TrilinearTemporalAligner).__init__()
        self.T = T
        self.vox_util = utils.vox.Vox_util(nZ, nY, nX, bounds)
        xyz_memA = utils.basic.gridcloud3d(1, nZ, nY, nX, norm=False)
        self.xyz_imu = torch.nn.Parameter(self.vox_util.Mem2Ref(xyz_memA, nZ, nY, nX, assert_cube=False), require_grad=False) # 1, nZ*nY*nX, 3
        self.imu2mem = torch.nn.Parameter(self.vox_util.get_mem_T_ref(1, nZ, nY, nX), require_grad=False) # 1, 4, 4
        self.output_mapper = MODELS.build(**output_mapper)

    def forward(self, feat3ds, global_poses):
        # T is sequence length
        # feat3d: B, T, C, nZ, nY, nX 
        # global_poses: B, T, 4, 4
        B, T, C, nZ, nY, nX = feat3ds.shape
        cur_pose = global_poses[:, -1, :, :] # B, 4, 4
        prev_poses = global_poses[:, :-1, :, :] #  B, T-1, 4, 4
        out_feat3ds = []
        miss_Ts = self.T - T
        # pad zeros for missing frames
        for t in range(miss_Ts):
            out_feat3ds.append(torch.zeros_like(feat3ds[:, 0]))
        # align previous frames to current frame
        for t in range(self.T - 1 - miss_Ts):
            prev_pose = prev_poses[:, t, :, :] # B, 4, 4
            cur2prev = torch.matmul(utils.geom.safe_inverse(prev_pose), cur_pose)  # B, 4, 4
            cur2prev_mem = torch.matmul(self.imu2mem.repeat(B, 1, 1), cur2prev) # B, 4, 4
            self.xyz_prev_mem = utils.geom.apply4x4(cur2prev_mem, self.xyz_imu.repeat(B, 1, 1)) # (B, 4, 4) x (B, nZ*nY*nX, 3) -> (B, nZ*nY*nX, 3)
            feat3d = feat3ds[:, t] # B, C, nZ, nY, nX
            feat3d = torch.nn.functional.grid_sample(feat3d, self.xyz_prev_mem.view(B, nZ, nY, nX, 3)) # B, C, nZ, nY, nX
            out_feat3ds.append(feat3d)
        # append current frame
        cur_feat3d = feat3ds[:, -1] 
        out_feat3ds.append(cur_feat3d)
        out_feat3ds = torch.cat(out_feat3ds, dim=1) # B,  T*C, nZ, nY, nX
        return self.output_mapper(out_feat3ds)

