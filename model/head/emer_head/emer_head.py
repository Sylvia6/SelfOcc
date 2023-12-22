from ..base_head import BaseTaskHead
from mmengine.registry import MODELS

@MODELS.register_module()
class EmerNerfHead(BaseTaskHead):
    def __init__(self, init_cfg=None, **kwargs):
        super().__init__(init_cfg, **kwargs)


    def __init__(self, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        self.static_mapper = MODELS.build(**static_mapper)
        self.dynamic_mapper = MODELS.build(**dynamic_mapper)
    
    def get_feats(self, feat3ds):
        return (self.static_mapper(feat3ds), self.dynamic_mapper(feat3ds))

    def forward(
        self, 
        representation,
        points=None,
        metas=None,
        **kwargs
    ):
        # representation: B, C, nZ, nY, nX
        static_feat, dynamic_feat = self.get_feats(representation)
        # B, C, nZ, nY, nX
        # B, C, nT, nZ, nY, nX