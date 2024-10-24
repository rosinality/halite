from vllm.worker import worker


class Worker(worker.Worker):
    def __init__(self, model, model_config, parallel_config, scheduler_config):
        super().__init__(model_config, parallel_config, scheduler_config)

        self.model = model

    def init_model(self):
        pass
