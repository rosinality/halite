from sglang.srt.model_executor import model_runner


class ModelRunner(model_runner.ModelRunner):
    def init_torch_distributed(self):
        pass

    def load_model(self):
        pass
