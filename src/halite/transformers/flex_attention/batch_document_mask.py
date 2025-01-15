import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature

from halite.transformers.flex_attention import FlexAttentionUpdateMode


def _batch_offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[:, 1:] - offsets[:, :-1]
    batch_size = offsets.shape[0]

    return torch.repeat_interleave(
        torch.arange(counts.shape[1], device=device, dtype=torch.int32).repeat(
            batch_size
        ),
        counts.reshape(-1),
    ).reshape(batch_size, -1)


def batch_length_to_offsets(lengths: list[int], device: str | torch.device) -> Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets_raw = torch.as_tensor(lengths, device=device, dtype=torch.int32)
    offsets = torch.zeros(
        (lengths.shape[0], lengths.shape[1] + 1), device=device, dtype=torch.int32
    )
    offsets[:, 1:] = offsets_raw
    offsets = torch.cumsum(offsets, -1)

    return offsets


class BatchDocumentMask:
    inputs: tuple[str] = ("document_offsets",)
    update_mode: FlexAttentionUpdateMode = FlexAttentionUpdateMode.BATCH
    head_shared: bool = True

    def __init__(self, mask_mod: _mask_mod_signature):
        # get mask mod function, currently does not support mod function with args
        self.mask_mod_fn = mask_mod()

    def __call__(self, document_offsets: Tensor):
        """Generates mask mods that apply to inputs to flex attention in the sequence stacked
        format.

        Args:
            mask_mod: The mask mod to apply to the documents
            offsets: This tensor should be of shape(num_documents + 1)
                this should contain the cumulative counts of document tokens.
                e.g. if you have 3 documents of length 2, 4, 3 then
                offsets = [0, 2, 6, 9]

        Note:
            What is the sequence stacked format? When assembling batches of inputs, we
            take multiple sequences and stack them together to form 1 large sequence. We then
            use masking to ensure that the attention scores are only applied to tokens within
            the same document.
        """
        document_id = _batch_offsets_to_doc_ids_tensor(document_offsets)

        def doc_mask_mod(b, h, q_idx, kv_idx):
            same_doc = document_id[b, q_idx] == document_id[b, kv_idx]
            q_logical = q_idx - document_offsets[b, document_id[b, q_idx]]
            kv_logical = kv_idx - document_offsets[b, document_id[b, kv_idx]]
            inner_mask = self.mask_mod_fn(b, h, q_logical, kv_logical)

            return same_doc & inner_mask

        return doc_mask_mod
