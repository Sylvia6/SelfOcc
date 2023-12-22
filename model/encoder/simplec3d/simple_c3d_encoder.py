from mmseg.registry import MODELS
from ..base_encoder import BaseEncoder


@MODELS.register_module()
class SimpleC3DEncoder(BaseEncoder):
    def __init__(
        self,
        input_mapper,
        temporal_aligner,
        neck3d,
        init_cfg=None):
        super().__init__(init_cfg)
        self.input_mapper = MODELS.build(**input_mapper)
        self.temporal_aligner = MODELS.build(**temporal_aligner)
        self.neck3d = MODELS.build(**neck3d)

    def forward(
        self, 
        representation,
        ms_img_feats=None,
        metas=None,
        **kwargs):
        # representation: [B, nT, C, nZ, nY, nX]
        fea3ds = self.input_mapper(representation.reshape(-1, *representation.shape[2:])) # B*nT, C, nZ, nY, nX
        fea3ds = fea3ds.reshape(*representation.shape[:2], -1, *fea3ds.shape[1:]) # B, nT, C, nZ, nY, nX

        # temporal align
        global_poses = metas["input"]["imu2world"].to(representation.device) # [B, nT, 4, 4]
        fea3ds = self.temporal_aligner(fea3ds, global_poses) # [B, C, nZ, nY, nX]

        # 3d neck
        fea3ds = self.neck3d(fea3ds) # [B, C, nZ*s, nY*s, nX*s] TODO return a list or a dict
        return fea3ds
