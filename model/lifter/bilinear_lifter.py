import torch

from mmengine.registry import MODELS
from mmengine.model import BaseModule

from .. import utils


@MODELS.register_module()
class BilinearLifter(BaseModule):
    def __init__(self, inc, N_cam, nZ, nY, nX, bounds, embed_cam=False):
        super(BilinearLifter, self).__init__()

        # lift coords
        self.vox_util = utils.vox.Vox_util(nZ, nY, nX, bounds)
        xyz_memA = utils.basic.gridcloud3d(1, nZ, nY, nX, norm=False)
        self.xyz_imu = torch.nn.Parameter(self.vox_util.Mem2Ref(xyz_memA, nZ, nY, nX, assert_cube=False), require_grad=False) # 1, nZ*nY*nX, 3

        # cam embbedding
        self.inc = inc
        self.N_cam = N_cam
        self.embed_cam = embed_cam
        if self.embed_cam:
            self.cam_embedder = torch.nn.Embedding(N_cam, inc)

    def forward(self, feat2ds, cam2imus, cam2imgs, imgH, imgW):
        # Note that the B can be B x S, where S is the sequence length, but we don't need to care about it here.
        # feat2d: [B, N, C, H, W]
        # cam2imus: [B, N, 4, 4]  aka camera pose
        # cam2imgs: [B, N, 4, 4]  aka camera intrinsics
        # imgHs:   image heights
        # imgWs:   image widths
        # Terminology:
        #  imu means the coordinate in a frame
        #  img means the original img
        #  feat2d means the 2d feature map of the img after backbone and neck
        #  mem means memory space, coordinate start from 0 to nZ*nY*nX -1
        B, N, C, H, W = feat2ds.shape
        assert N == self.N_cam
        assert C == self.inc
        if self.embed_cam:
            embedding = self.cam_embedder(torch.arange(N, device=feat2ds.device).unsqueeze(0).repeat(B, 1)) # B, N, C
            feat2ds = feat2ds + embedding.unsqueeze(-1).unsqueeze(-1)

        # lift
        feat2ds = utils.basic.pack_seqdim(feat2ds, B) # B*N, C, H, W
        cam2imus = utils.basic.pack_seqdim(cam2imus, B) # B*N, 4, 4
        imu2cams = utils.geom.safe_inverse(cam2imus) # B*N, 4, 4
        cam2imgs = utils.basic.pack_seqdim(cam2imgs, B) # B*N, 4, 4
        # TODO to make the torchscript happy, we need to predefine the shape of the feat2d
        sy = H / float(imgH)
        sx = W / float(imgW)
        cam2feat2ds = utils.geom.scale_intrinsics(cam2imgs, sx, sy)
        # the code in the unproject_image_to_mem is not explicitly, but it is correct 
        feat3ds = self.vox_util.unproject_image_to_mem(
            cam2feat2ds,
            utils.basic.matmul2(cam2feat2ds, imu2cams),
            imu2cams,
            xyz_camA=self.xyz_imu) # B*N, C, nZ, nY, nX
        feat3ds = utils.basic.unpack_seqdim(feat3ds, B) # B, N, C, nZ, nY, nX
        if self.embed_cam:
            feat3ds = feat3ds.sum(dim=1) # B, C, nZ, nY, nX
        else: # concat
            feat3ds = feat3ds.view(B, N*C, self.vox_util.Z, self.vox_util.Y, self.vox_util.X) # B, N*C, nZ, nY, nX
        return feat3ds