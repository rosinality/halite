class SFTSample:
    def __init__(
        self,
        prompt_key,
        response_key,
        tokenizer,
        input_key="input",
        target_key="target",
        prompt_map_fn=None,
        response_map_fn=None,
        prompt_tokenizer_kwargs=None,
        response_tokenizer_kwargs=None,
        ignore_index=-100,
    ):
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.target_key = target_key
        self.prompt_map_fn = prompt_map_fn
        self.response_map_fn = response_map_fn
        self.prompt_tokenizer_kwargs = (
            prompt_tokenizer_kwargs if prompt_tokenizer_kwargs is not None else {}
        )
        self.response_tokenizer_kwargs = (
            response_tokenizer_kwargs if response_tokenizer_kwargs is not None else {}
        )
        self.ignore_index = ignore_index

    def __call__(self, iterator):
        for record in iterator:
            yield record

    def process(self, record):
        prompt = record[self.prompt_key]
        response = record[self.response_key]

        if self.prompt_map_fn is not None:
            prompt = self.prompt_map_fn(prompt)

        if self.response_map_fn is not None:
            response = self.response_map_fn(response)

        prompt_tokens = self.tokenizer(
            prompt,
            **self.prompt_tokenizer_kwargs,
        )

        response_tokens = self.tokenizer(
            response,
            **self.response_tokenizer_kwargs,
        )

        all_tokens = prompt_tokens + response_tokens
        target = all_tokens.copy()
        target[: -len(response_tokens)] = self.ignore_index

        record[self.input_key] = all_tokens[:-1]
        record[self.target_key] = target[1:]

        return record
