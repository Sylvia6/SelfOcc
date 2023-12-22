from typing import Literal
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from projects.emernerf.emer_pipeline import EmerPipeline


class EmerTrainer(Trainer):
    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config, local_rank, world_size)

    def setup(self, test_mode: Literal['test', 'val', 'inference'] = "val") -> None:
        result =  super().setup(test_mode)
        if isinstance(self.pipeline, EmerPipeline):
            self.pipeline.set_total_iters(self.config.max_num_iterations)
        return result

