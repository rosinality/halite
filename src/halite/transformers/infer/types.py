class InferenceResult:
    def __init__(
        self,
        id: int | str,
        input_ids: list[int] | None = None,
        output_ids: list[list[int]] | None = None,
        logprobs: list[list[float]] | None = None,
        finish_reason: list[str] | None = None,
        num_cached_prompt_tokens: list[int] | None = None,
    ):
        self.id = id

        self.input_ids = [] if input_ids is None else input_ids
        self.output_ids = [] if output_ids is None else output_ids

        self.logprobs = [] if logprobs is None else logprobs
        self.finish_reason = [] if finish_reason is None else finish_reason
        self.num_cached_prompt_tokens = (
            [] if num_cached_prompt_tokens is None else num_cached_prompt_tokens
        )

    def validate_lengths(self):
        if not (
            len(self.output_ids)
            == len(self.logprobs)
            == len(self.finish_reason)
            == len(self.num_cached_prompt_tokens)
        ):
            raise ValueError(
                f"lengths of output_ids, logprobs, finish_reason, and num_cached_prompt_tokens must be equal. Got {len(self.output_ids)}, {len(self.logprobs)}, {len(self.finish_reason)}, and {len(self.num_cached_prompt_tokens)}"
            )

    def to_dict(self):
        return {
            "id": self.id,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
        }

    def __repr__(self):
        if len(self.output_ids) > 2:
            output_ids = [self.output_ids[0], "...", self.output_ids[-1]]

        else:
            output_ids = self.output_ids

        output_ids = ", ".join(str(out) for out in output_ids)

        return f"InferenceResult(id={self.id}, input_ids={self.input_ids}, output_ids=[{output_ids}])"
