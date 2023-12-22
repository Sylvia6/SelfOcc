from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.method_configs import all_methods, all_descriptions
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from projects.emernerf.emer_pipeline import EmerPipeline
from projects.emernerf.plus_dataparser import PlusDataParserConfig
from projects.emernerf.plus_dataset import PlusDataset
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from projects.emernerf.emernerf import EmerNerfModelConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
import tyro


emernerf_iters = 10000
all_methods["emernerf"] = TrainerConfig(
    method_name="emernerf",
    steps_per_eval_batch=500,
    steps_per_save=5000,
    max_num_iterations=emernerf_iters,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        _target = EmerPipeline,
        datamanager=VanillaDataManagerConfig(
            dataparser=PlusDataParserConfig(),
            _target=VanillaDataManager[PlusDataset],
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=EmerNerfModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            num_iters=emernerf_iters
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]