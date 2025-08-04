import re

import torch


def get_answer(predict_str: str) -> str:
    pattern = re.compile(r"<answer>(.*)</answer>", re.DOTALL)
    match_result = re.search(pattern, predict_str)

    return match_result.group(1) if match_result else "None"


class MathVerify:
    def __init__(self, timeout_score=0, multiprocess_mode=False, timeout_seconds=5):
        self.timeout_score = timeout_score

        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        if multiprocess_mode:
            from halite.projects.rewards.math_verify import math_metric, timeout

            verify_fn = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ),
            )

            self.verify_fn = timeout(timeout_seconds)(verify_fn)

        else:
            from math_verify.metric import math_metric

            verify_fn = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ),
            )

            self.verify_fn = verify_fn

    def verify(self, model_output, ground_truth):
        from math_verify.errors import TimeoutException

        score = 0.0

        model_output = get_answer(model_output)

        if model_output == "None":
            return score

        model_output = f"\\boxed{{{model_output}}}"

        try:
            score, _ = self.verify_fn([ground_truth], [model_output])

        except Exception as e:
            print(e)

            pass

        except TimeoutException:
            score = self.timeout_score

        return score

    def __call__(self, model_output, ground_truth):
        rewards = []

        for out, truth in zip(model_output, ground_truth):
            rewards.append(self.verify(out, truth))

        return torch.tensor(rewards)


class FormatReward:
    def check_format(self, model_output):
        pattern = re.compile(r".*</think>.*<answer>.*</answer>.*", re.DOTALL)

        match_result = re.fullmatch(pattern, model_output)

        return 1.0 if match_result else 0.0

    def __call__(self, model_output):
        rewards = []

        for out in model_output:
            rewards.append(self.check_format(out))

        return torch.tensor(rewards)
