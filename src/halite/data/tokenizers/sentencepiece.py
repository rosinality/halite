from sentencepiece import SentencePieceProcessor


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.processor = SentencePieceProcessor(model_path)

        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()

    def encode(self, text: str, *, bos: bool, eos: bool) -> list[int]:
        tokens = self.processor.encode(text)

        if bos:
            tokens.insert(0, self.bos_id)

        if eos:
            tokens.append(self.eos_id)

        return tokens
