
import torch

from mmengine.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class SimpleSegementor(BaseModule):
    def __init__(self,
            freeze_img_backbone=False,
            freeze_img_neck=False,
            backbone, neck, lifter, encoder, head):
        super(SimpleSegementor, self).__init__()
        self.img_backbone = MODELS.build(**backbone)
        self.img_neck = MODELS.build(**neck)
        self.lifter = MODELS.build(**lifter)
        self.encoder = MODELS.build(**encoder)
        self.head = MODELS.build(**head)

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)


    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                img_feat_only=False,
                extra_backbone=False,
                occ_only=False,
                prepare=False,
                **kwargs,
        ):
        # imgs: [B, nT, N, 3, H, W]
        # Batch, Sequence, Num of cameras, rgb channels, Height, Width

        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points
        }

        B, nT, N, C, H, W = imgs.shape
        imgs = imgs.view(B*nT*N, C, H, W)
        fea2ds=self.img_neck(self.img_backbone(imgs) # [B*nT*N, C, H, W]
        # 2d -> 3d
        _, C, H, W = fea2ds.shape
        fea3ds = self.lifter(fea2ds.view(B*nT, N, C, H, W), metas) # [B*nT, C, nZ, nY, nX]
        # 3d -> 4d -> dynamic_3d 
        _, C, nZ, nY, nX = fea3ds.shape
        fea3ds = fea3ds.view(B, nT, C, nZ, nY, nX)
        fea3ds = self.encoder(fea3ds, metas = metas)
        results["representation"] = fea3ds

        # get rendered result
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        elif prepare and hasattr(self.head, "prepare"):
            outs = self.head.prepare(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return self.nerf_head(fea3ds, metas)