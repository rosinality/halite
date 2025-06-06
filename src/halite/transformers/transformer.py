from dataclasses import dataclass
import time
from typing import Any, List, Optional

import torch
from torch import nn, utils

from slickconf import Field

from halite.transformers.cache_manager import AllocatedCacheManager
from halite.transformers.container import ModuleDict
from halite.transformers.generation import GenerationMixin
from halite.transformers.model import ModelMixin


@dataclass
class TransformerDecoderOutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    loss_dict: Optional[dict[str, torch.Tensor]] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None
    cache: Optional[Any] = None
    aux_outputs: Optional[Any] = None


class TransformerConfig(Field):
    pass


class TransformerDecoder(nn.Module, GenerationMixin, ModelMixin):
    def __init__(
        self,
        embedding,
        post_embed=None,
        attention_mask=None,
        pos_embed=None,
        blocks=None,
        post_blocks=None,
        head=None,
        tie_embeds=False,
        use_position_ids=False,
        attention_processor="native",
        flex_attention_processor=None,
        config=None,
    ):
        super().__init__()

        self.embedding = embedding
        self.post_embed = post_embed

        self.attention_mask = attention_mask

        self.pos_embed = pos_embed
        self.pos_embed_attention_bias = getattr(pos_embed, "attention_bias", False)
        self.pos_embed_layer_shared = getattr(pos_embed, "layer_shared", True)

        self.blocks = ModuleDict()
        for i, block in enumerate(blocks):
            self.blocks[str(i)] = block

        self.post_blocks = post_blocks
        self.head = head

        self.use_position_ids = use_position_ids

        self.flex_attention_processor = flex_attention_processor

        self.tie_embeds = tie_embeds
        if tie_embeds and self.head is not None:
            self.head.tie_weight(self.embedding.embed_weight)

        self.gradient_checkpointing = 0

        self.adapters = None
        self.cache_manager = None

        self.config = config

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        if self.tie_embeds:
            del state_dict["head." + self.head.tie_weight_key]

        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.tie_embeds:
            state_dict["head." + self.head.tie_weight_key] = state_dict[
                "embedding." + self.embedding.tie_weight_key
            ]

        super().load_state_dict(state_dict, strict, assign)

    def to_empty(self, *args, **kwargs):
        super().to_empty(*args, **kwargs)

        if self.tie_embeds and self.head is not None:
            self.head.tie_weight(self.embedding.embed_weight)

    def init_weights(self, device):
        def init_weight(module):
            if hasattr(module, "init_weights"):
                module.init_weights()

        for child in self.children():
            child.apply(init_weight)

        self.init_buffers(device)

    def init_buffers(self, device):
        def init_buffer(module):
            if hasattr(module, "init_buffers"):
                module.init_buffers(device)

        for child in self.children():
            child.apply(init_buffer)

    def add_adapter(self, name, keys):
        if self.adapters is None:
            self.adapters = {}

        self.adapters[name] = keys

    def init_cache_manager(self, manager, **kwargs):
        if manager == "allocated":
            param = next(iter(self.parameters()))

            device = kwargs.get("device", param.device)
            dtype = kwargs.get("dtype", param.dtype)

            self.cache_manager = AllocatedCacheManager(
                n_cache=len(self.blocks),
                n_head=self.config.n_head,
                head_dim=self.config.head_dim,
                device=device,
                dtype=dtype,
                **kwargs,
            )

    def reset_cache_manager(self):
        self.cache_manager.reset()

    def gradient_checkpointing_enable(self, n_layer=1):
        self.gradient_checkpointing = n_layer

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._require_grads_hook = (
            self.embedding.get_input_embeddings().register_forward_hook(
                make_inputs_require_grads
            )
        )

    def disable_input_require_grads(self):
        self._require_grads_hook.remove()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache=None,
        use_cache=False,
        attention_mask=None,
        position_ids=None,
        slice_index=None,
        unfinished=None,
        **kwargs,
    ):
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if (
            self.use_position_ids
            and attention_mask is not None
            and position_ids is None
        ):
            position_ids = attention_mask.to(torch.int64).cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            if cache is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if unfinished is not None:
            # for synced_gpus=True, at least 1 sequences should be forwarded
            if unfinished.max() == 0:
                unfinished = unfinished.clone()
                unfinished[0] = 1

            unfinished = unfinished.to(torch.bool)
            input_ids = input_ids[unfinished]
            attention_mask = attention_mask[unfinished]

            if position_ids is not None:
                position_ids = position_ids[unfinished]

        if slice_index is not None and cache is not None:
            cache.select(slice_index)

        return {
            "input_ids": input_ids,
            "cache": cache,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **kwargs,
        }

    def get_pos_embed(
        self, pos_embed, attention_mask, query_length, position_ids, device, dtype
    ):
        if self.pos_embed_attention_bias:
            out = pos_embed(attention_mask)

        else:
            out = pos_embed(position_ids, query_length, device, dtype)

        return out

    def get_attention_mask(
        self, attention_mask, batch, query_length, key_length, device
    ):
        attention_mask_candid = None
        if self.attention_mask is not None:
            attention_mask_candid = self.attention_mask(
                batch, query_length, key_length, device
            )

        if attention_mask.ndim < 4:
            expand_mask = ~attention_mask[:, None, None, :].to(torch.bool)
            expand_mask = expand_mask.expand(batch, 1, query_length, key_length)

        attention_mask = (
            expand_mask
            if attention_mask_candid is None
            else expand_mask | attention_mask_candid
        )

        return attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        images=None,
        cache=None,
        use_cache=False,
        output_hidden_states=False,
        bidirection_ids=None,
        document_offsets=None,
        target=None,
        unpad_params=None,
    ):
        embedding_input = {}

        if input_ids is not None:
            embedding_input["input_ids"] = input_ids

        if images is not None:
            embedding_input["images"] = images

        out = self.embedding(**embedding_input)

        if self.post_embed is not None:
            out = self.post_embed(out)

        if cache is None:
            cache = [None] * len(self.blocks)

        query_length = out.shape[1]

        key_length = query_length
        past_key_length = 0

        if cache[0] is not None:
            past_key_length = cache[0].length
            key_length += past_key_length

        if self.use_position_ids:
            if position_ids is None:
                position_ids = torch.arange(
                    past_key_length,
                    query_length + past_key_length,
                    device=out.device,
                )
                position_ids = position_ids.unsqueeze(0)

        pos_emb = None
        if self.pos_embed is not None and self.pos_embed_layer_shared:
            pos_emb = self.get_pos_embed(
                self.pos_embed,
                attention_mask,
                query_length,
                position_ids,
                out.device,
                out.dtype,
            )

        if attention_mask is not None:
            attention_mask = self.get_attention_mask(
                attention_mask,
                input_ids.shape[1],
                input_ids.shape[0],
                key_length,
                input_ids.device,
            )

        if self.flex_attention_processor is not None:
            attention_mask = self.flex_attention_processor(
                batch=input_ids.shape[0],
                q_len=query_length,
                kv_len=key_length,
                bidirection_ids=bidirection_ids,
                position_ids=position_ids,
                document_offsets=document_offsets,
            )

        next_caches = []
        aux_outputs = []

        hidden = []
        residual = None

        if self.gradient_checkpointing > 0 and self.training:
            for index in list(range(len(self.blocks)))[:: self.gradient_checkpointing]:

                def create_custom_forward(modules):
                    def custom_forward(*inputs):
                        out, residual, *rest = inputs

                        for module in modules:
                            out, residual, _ = module(out, residual, *rest)

                        return out, residual

                    return custom_forward

                # attention_bias, pos_emb, cache, use_cache, unpad_params
                rest_inputs = [
                    None,
                    pos_emb,
                    None,
                    False,
                    unpad_params,
                ]

                out, residual = utils.checkpoint.checkpoint(
                    create_custom_forward(
                        [
                            self.blocks[str(index)]
                            for index in range(
                                index, index + self.gradient_checkpointing
                            )
                        ]
                    ),
                    out,
                    residual,
                    attention_mask,
                    *rest_inputs,
                    use_reentrant=False,
                )

        else:
            for index, block in self.blocks.items():
                index = int(index)

                if output_hidden_states:
                    hidden.append(out)

                if not self.pos_embed_layer_shared:
                    pos_emb = self.get_pos_embed(self.pos_embeds[index], attention_mask)

                out, residual, next_cache, aux_out = block(
                    out,
                    residual,
                    attention_mask,
                    pos_emb=pos_emb,
                    use_cache=use_cache,
                    cache=cache[index],
                    unpad_params=unpad_params,
                )

                if use_cache:
                    next_caches.append(next_cache)

                aux_outputs.append(aux_out)

        if self.post_blocks is not None:
            if residual is not None:
                out = self.post_blocks(out, residual)

            else:
                out = self.post_blocks(out)

        if output_hidden_states:
            hidden.append(out)

        if self.head is None:
            return TransformerDecoderOutput(
                last_hidden_state=out,
                hidden_states=hidden,
                cache=next_caches,
                aux_outputs=aux_outputs,
            )

        if target is not None:
            out, loss_dict = self.head(out, target)

            return TransformerDecoderOutput(
                loss=out,
                loss_dict=loss_dict,
                hidden_states=hidden,
                cache=next_caches,
                aux_outputs=aux_outputs,
            )

        else:
            out = self.head(out)

            return TransformerDecoderOutput(
                logits=out,
                hidden_states=hidden,
                cache=next_caches,
                aux_outputs=aux_outputs,
            )
