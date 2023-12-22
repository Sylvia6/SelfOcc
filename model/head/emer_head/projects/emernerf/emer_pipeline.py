from nerfstudio.pipelines.base_pipeline import VanillaPipeline


class EmerPipeline(VanillaPipeline):
    def __init__(
        self,
        *arg,
        **kwargs
    ):
        super().__init__(*arg, **kwargs)
        # self._model
        timestamps, time_diff = self.datamanager.train_dataset.get_training_unique_timestamps()
        self._model.register_normalized_training_timesteps(timestamps, time_diff)

